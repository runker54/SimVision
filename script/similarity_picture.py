import concurrent.futures
import logging
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
from multiprocessing import Queue, Manager
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import cv2
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from scipy.fftpack import dct
from scipy.spatial.distance import cosine
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms, models
from tqdm import tqdm
from urllib3.util.retry import Retry

class Config:
    # 图像处理参数
    IMAGE_SIZE = (224, 224)
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]

    # 模型参数
    DEFAULT_MODEL_NAMES = ['resnet50']
    DEFAULT_METHODS = ['histogram', 'hash', 'mse', 'template', 'phash']

    # 相似度阈值
    DEFAULT_THRESHOLD = 0.9

    # 并行处理参数
    DEFAULT_NUM_WORKERS = 48
    DEFAULT_BATCH_SIZE = 50

    # 缓存参数
    CACHE_SIZE = 1000

    @classmethod
    def get_image_size(cls, model_name: str) -> Tuple[int, int]:
        sizes = {
            'inception_v3': (299, 299),
            'efficientnet_b0': (224, 224),
            # 可以为其他模型添加特定的大小
        }
        return sizes.get(model_name, cls.IMAGE_SIZE)

# 设置全局设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_start_method():
    """
    设置多进程的启动方法。
    如果CUDA可用，或者在Windows系统上，使用'spawn'方法；
    否则在其他系统上使用'fork'方法。
    """
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    else:
        if sys.platform.startswith('win'):
            multiprocessing.set_start_method('spawn', force=True)
        else:
            multiprocessing.set_start_method('fork', force=True)

def setup_logger(log_file_path):
    """设置日志记录器。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def is_url(path: str) -> bool:
    """判断给定的路径是否为URL。"""
    return path.startswith('http://') or path.startswith('https://')

def fix_url(url: str) -> str:
    """修复URL中的反斜杠。"""
    return url.replace('\\', '/')

def encode_path(path: str) -> str:
    """对路径进行编码。在Windows系统上使用GBK编码，其他系统不变。"""
    if sys.platform.startswith('win'):
        return path.encode('gbk').decode('gbk')
    return path

class ImageManager:
    """图像管理类，用于加载和缓存图像。"""
    def __init__(self, cache_size=Config.CACHE_SIZE):
        self.images = {}
        self.resized_images = {}
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=0.1)
        self.session.mount('http://', HTTPAdapter(max_retries=retry))
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        self.cache_size = cache_size
        self.image_lru = []

    @lru_cache(maxsize=100)
    def load_image(self, img_path: str) -> np.ndarray:
        """加载图像并缓存。"""
        try:
            if img_path not in self.images:
                img = self.safe_read_image(img_path)
                if img is not None:
                    self.images[img_path] = img
                    self.image_lru.append(img_path)
                    if len(self.image_lru) > self.cache_size:
                        oldest = self.image_lru.pop(0)
                        del self.images[oldest]
            return self.images.get(img_path)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None

    def load_resized_image(self, img_path: str, size: Tuple[int, int]) -> np.ndarray:
        """加载并调整图像大小。"""
        key = (img_path, size)
        if key not in self.resized_images:
            original = self.load_image(img_path)
            if original is not None:
                resized = cv2.resize(original, size)
                self.resized_images[key] = resized
            else:
                return None
        return self.resized_images[key]

    def safe_read_image(self, img_path: str) -> np.ndarray:
        """安全地读取图像，支持本地文件和URL。"""
        try:
            if is_url(img_path):
                response = self.session.get(img_path, stream=True)
                response.raise_for_status()
                img_stream = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    img_stream.write(chunk)
                img_stream.seek(0)
                img_array = np.frombuffer(img_stream.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            else:
                img_path = os.path.normpath(os.path.abspath(img_path))
                with open(img_path, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError(f"Unable to decode image: {img_path}")
            return img
        except Exception as e:
            logging.error(f"Error reading image {img_path}: {e}")
            logging.error(f"File path: {img_path}")
            if not is_url(img_path):
                logging.error(f"File exists: {os.path.exists(img_path)}")
                logging.error(f"File size: {os.path.getsize(img_path) if os.path.exists(img_path) else 'N/A'}")
            return None

    def clear_cache(self):
        """清除图像缓存。"""
        self.images.clear()
        self.resized_images.clear()
        self.image_lru.clear()

image_manager = ImageManager()

class SiameseFeatureExtractor:
    """孪生网络特征提取器类。"""
    def __init__(self, model_name: str = 'resnet50', logger: logging.Logger = None):
        self.model_name = model_name
        self.image_size = Config.get_image_size(model_name)
        self.model = self._get_model()
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.logger = logger
        
        self.preprocess = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZATION_MEAN, std=Config.NORMALIZATION_STD),
        ])

    def _get_model(self):
        """获取预训练模型。"""
        model_configs = {
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
            'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
            'densenet121': (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
            'inception_v3': (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
            'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
            'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            'resnext50_32x4d': (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
            'resnext101_32x8d': (models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
            'wide_resnet50_2': (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
            'wide_resnet101_2': (models.wide_resnet101_2, models.Wide_ResNet101_2_Weights.IMAGENET1K_V1),
        }

        if self.model_name not in model_configs:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model_func, weights = model_configs[self.model_name]
        model = model_func(weights=weights)

        # Remove the last fully connected layer
        if self.model_name.startswith(('resnet', 'resnext', 'wide_resnet', 'inception')):
            model.fc = torch.nn.Identity()
        elif self.model_name.startswith(('mobilenet', 'efficientnet', 'vgg', 'densenet')):
            model.classifier = torch.nn.Identity()

        return model

    @torch.no_grad()
    def extract_features(self, img_path: str) -> np.ndarray:
        """从图像中提取特征。"""
        try:
            img = image_manager.load_resized_image(img_path, self.image_size)
            if img is None:
                raise ValueError(f"Unable to load image: {img_path}")
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image = self.preprocess(image).unsqueeze(0).to(DEVICE)
            features = self.model(image)
            return features.cpu().numpy().flatten()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unable to extract image features: {e}")
            return None

    def __del__(self):
        """析构函数，用于释放GPU内存。"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

class MultiModelFeatureExtractor:
    """多模型特征提取器类。"""
    def __init__(self, model_names: List[str], logger: logging.Logger = None):
        self.extractors = {name: SiameseFeatureExtractor(name, logger) for name in model_names}
        self.logger = logger

    def extract_features(self, img_path: str) -> Dict[str, np.ndarray]:
        """使用多个模型从图像中提取特征。"""
        features = {}
        for name, extractor in self.extractors.items():
            features[name] = extractor.extract_features(img_path)
        return features

def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """计算两个特征向量之间的余弦相似度。"""
    return cosine_similarity([feat1], [feat2])[0][0]

def compute_multi_model_similarity(feat1: Dict[str, np.ndarray], feat2: Dict[str, np.ndarray]) -> Dict[str, float]:
    """计算多个模型提取的特征之间的相似度。"""
    similarities = {}
    for model_name in feat1.keys():
        if feat1[model_name] is not None and feat2[model_name] is not None:
            similarities[model_name] = compute_cosine_similarity(feat1[model_name], feat2[model_name])
        else:
            similarities[model_name] = None
    return similarities



def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的直方图相似度。"""
    try:
        if img1 is None or img2 is None:
            return None
        
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        return 1 - cosine(hist1, hist2)
    except Exception as e:
        logging.error(f"Error calculating histogram similarity: {e}")
        return None


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的结构相似性（SSIM）。"""
    try:
        if img1 is None or img2 is None:
            return None
        
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        height, width = img1_gray.shape
        img2_gray = cv2.resize(img2_gray, (width, height))
        
        return ssim(img1_gray, img2_gray)
    except Exception as e:
        logging.error(f"Error calculating SSIM: {e}")
        return None


def calculate_hash_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的哈希相似度。"""
    def calculate_average_hash(img):
        """计算图像的平均哈希值。"""
        try:
            if img is None:
                return None
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (8, 8))
            mean = np.mean(img_resized)
            hash_value = 0
            for i in range(8):
                for j in range(8):
                    if img_resized[i, j] > mean:
                        hash_value |= 1 << (i * 8 + j)
            return hash_value
        except Exception as e:
            logging.error(f"计算平均哈希值时出错: {e}")
            return None

    hash1 = calculate_average_hash(img1)
    hash2 = calculate_average_hash(img2)
    
    if hash1 is None or hash2 is None:
        return None
    
    hamming_distance = bin(hash1 ^ hash2).count('1')
    return 1 - hamming_distance / 64.0


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的均方误差（MSE）。"""
    try:
        if img1 is None or img2 is None:
            return None
        # 确保两个图像具有相同的尺寸和通道数
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 如果通道数不同，将两个图像都转换为RGB格式
        if img1.shape[2] != img2.shape[2]:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB) if img1.shape[2] == 4 else img1
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB) if img2.shape[2] == 4 else img2
        
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        err = np.sum((img1 - img2) ** 2)
        err /= float(img1.shape[0] * img1.shape[1] * img1.shape[2])
        return err
    except Exception as e:
        logging.error(f"计算MSE时出错: {e}")
        return None


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的峰值信噪比（PSNR）。 """
    mse = calculate_mse(img1, img2)
    if mse is None or mse == 0:
        return None
    return 20 * np.log10(1.0) - 10 * np.log10(mse)


def calculate_template_matching(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的模板匹配相似度。"""
    try:
        if img1 is None or img2 is None:
            return None
        
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
        res = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
        return np.max(res)
    except Exception as e:
        logging.error(f"计算模板匹配时出错: {e}")
        return None


def calculate_glcm_features(img: np.ndarray) -> np.ndarray:
    """计算图像的灰度共生矩阵（GLCM）特征。 """
    try:
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        features = []
        for prop in properties:
            features.append(graycoprops(glcm, prop).flatten())
        
        return np.hstack(features)
    except Exception as e:
        logging.error(f"计算GLCM特征时出错: {e}")
        return None

def calculate_glcm_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的GLCM相似度。"""
    feat1 = calculate_glcm_features(img1)
    feat2 = calculate_glcm_features(img2)
    
    if feat1 is None or feat2 is None:
        return None
    
    try:
        return 1 - cosine(feat1, feat2)
    except Exception as e:
        logging.error(f"计算GLCM相似度时出错: {e}")
        return None
    

def calculate_phash(img: np.ndarray) -> np.ndarray:
    """计算图像的感知哈希（pHash）。"""
    try:
        if img is None:
            return None
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (32, 32))
        
        dct_result = dct(dct(img_resized, axis=0), axis=1)
        dct_low = dct_result[:8, :8]
        med = np.median(dct_low)
        
        return dct_low > med
    except Exception as e:
        logging.error(f"计算pHash时出错: {e}")
        return None

def calculate_phash_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图像的pHash相似度。"""
    hash1 = calculate_phash(img1)
    hash2 = calculate_phash(img2)
    
    if hash1 is None or hash2 is None:
        return None
    
    hamming_distance = np.sum(hash1 != hash2)
    return 1 - hamming_distance / (8 * 8)

def process_image_pair(img1_path: str, img2_path: str, group_name: str, feature_extractor: MultiModelFeatureExtractor, methods: List[str]) -> Dict:
    """处理一对图像,计算它们之间的相似度。"""
    logging.info(f"正在处理图像对: {img1_path} 和 {img2_path}")
    
    # 加载图像
    img1 = image_manager.load_image(img1_path)
    img2 = image_manager.load_image(img2_path)
    
    if img1 is None or img2 is None:
        logging.warning(f"无法加载图像: {img1_path} 或 {img2_path}")
        return None
    
    # 提取特征
    feat1 = feature_extractor.extract_features(img1_path)
    feat2 = feature_extractor.extract_features(img2_path)
    
    if not feat1 or not feat2:
        logging.warning(f"无法提取特征: {img1_path} 或 {img2_path}")
        return None
    
    # 计算深度学习模型的相似度
    model_similarities = compute_multi_model_similarity(feat1, feat2)
    
    # 初始化结果字典
    result = {
        'Group Name': group_name,
        'Image 1': img1_path,
        'Image 2': img2_path,
    }
    result.update({f'{model_name} Similarity': sim for model_name, sim in model_similarities.items()})
    
    # 定义传统图像处理方法的相似度计算函数
    similarity_functions = {
        'histogram': calculate_histogram_similarity,
        'ssim': calculate_ssim,
        'hash': calculate_hash_similarity,
        'mse': calculate_mse,
        'psnr': calculate_psnr,
        'template': calculate_template_matching,
        'glcm': calculate_glcm_similarity,
        'phash': calculate_phash_similarity
    }
    
    # 计算指定的传统方法的相似度
    for method in methods:
        if method in similarity_functions:
            result[f'{method.capitalize()} Similarity'] = similarity_functions[method](img1, img2)
    
    return result

def process_group(group_name: str, image_paths: List[str], threshold: float, model_names: List[str], methods: List[str], log_list: List, error_list: List) -> List[Dict]:
    """
    处理一个图像组,计算组内所有图像对之间的相似度。
    
    参数:
    group_name: 图像组的名称
    image_paths: 组内所有图像的路径列表
    threshold: 相似度阈值
    model_names: 要使用的深度学习模型名称列表
    methods: 要使用的传统图像处理方法列表
    log_list: 用于记录处理日志的共享列表
    error_list: 用于记录错误信息的共享列表
    
    返回:
    包含所有相似图像对结果的列表
    """
    try:
        feature_extractor = MultiModelFeatureExtractor(model_names)
        similar_pairs = []
        n = len(image_paths)
        
        # 遍历所有可能的图像对
        for i in range(n):
            for j in range(i + 1, n):
                result = process_image_pair(image_paths[i], image_paths[j], group_name, feature_extractor, methods)
                if result and any(result.get(f'{model_name} Similarity', 0) >= threshold for model_name in model_names):
                    similar_pairs.append(result)
        
        # 清理缓存
        image_manager.clear_cache()
        
        log_list.append(f"组 {group_name} 处理完成, 找到 {len(similar_pairs)} 对相似图像")
        return similar_pairs
    except Exception as e:
        error_list.append(f"处理组 {group_name} 时出错: {str(e)}")
        return []

def process_groups_in_batches(groups: Dict[str, List[str]], threshold: float, num_workers: int, model_names: List[str], methods: List[str], logger: logging.Logger, batch_size: int = 50) -> List[Dict]:
    """
    批量处理图像组,计算组内所有图像对之间的相似度。

    参数:
    groups: 包含所有图像组的字典,键为组名,值为图像路径列表
    threshold: 相似度阈值
    num_workers: 并行处理的工作进程数
    model_names: 要使用的深度学习模型名称列表
    methods: 要使用的传统图像处理方法列表
    logger: 日志记录器
    batch_size: 每批处理的图像组数量

    返回:
    包含所有相似图像对结果的列表
    """
    all_results = []
    group_items = list(groups.items())
    total_batches = (len(group_items) + batch_size - 1) // batch_size
    total_groups = len(group_items)

    with Manager() as manager:
        log_list = manager.list()
        error_list = manager.list()

        logger.info(f"开始处理 {total_groups} 个图像组,共 {total_batches} 批")
        for i in tqdm(range(0, len(group_items), batch_size), total=total_batches, desc="处理批次"):
            batch = dict(group_items[i:i+batch_size])
            current_batch = i//batch_size + 1
            logger.info(f"正在处理第 {current_batch}/{total_batches} 批,包含 {len(batch)} 个组")
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_group = {executor.submit(process_group, name, images, threshold, model_names, methods, log_list, error_list): name 
                                   for name, images in batch.items()}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_group), total=len(batch), desc=f"第 {current_batch} 批进度"):
                    try:
                        group_result = future.result()
                        all_results.extend(group_result)
                    except Exception as e:
                        group_name = future_to_group[future]
                        error_message = f"处理组 {group_name} 时出错: {str(e)}"
                        error_list.append(error_message)
                        logger.error(error_message)

            # 处理日志列表
            for log_message in log_list:
                logger.info(log_message)
            log_list[:] = []  # 清空列表
            
            # 处理错误列表
            for error_message in error_list:
                logger.error(error_message)
            error_list[:] = []  # 清空列表
            
            logger.info(f"第 {current_batch}/{total_batches} 批处理完成")

    logger.info(f"所有 {total_groups} 个图像组处理完成")
    return all_results

def save_results_to_excel(results: List[Dict], filename: str, logger: logging.Logger):
    """
    将相似图像对的结果保存到Excel文件。

    参数:
    results: 包含所有相似图像对结果的列表
    filename: 要保存的Excel文件名
    logger: 日志记录器
    """
    if not results:
        logger.info("未找到相似的图像对。")
        return
    
    df = pd.DataFrame(results)
    df['Group Name'] = df['Group Name'].astype(str)
    df.to_excel(filename, index=False, engine='openpyxl')
    logger.info(f"结果已保存到 {filename}")

def setup_environment(output_dir: str, log_file: str) -> logging.Logger:
    """
    设置输出目录和日志记录器。

    参数:
    output_dir: 输出目录路径
    log_file: 日志文件名

    返回:
    配置好的日志记录器
    """
    os.makedirs(output_dir, exist_ok=True)
    return setup_logger(os.path.join(output_dir, log_file))

def normalize_image_groups(image_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    规范化图像组中的路径。

    参数:
    image_groups: 包含所有图像组的字典,键为组名,值为图像路径列表

    返回:
    规范化后的图像组字典
    """
    return {
        group_name: [fix_url(path) if is_url(path) else encode_path(os.path.normpath(path)) for path in paths]
        for group_name, paths in image_groups.items()
    }

def log_analysis_settings(logger: logging.Logger, normalized_groups: Dict[str, List[str]], model_names: List[str], methods: List[str], threshold: float, num_workers: int, batch_size: int):
    """
    记录分析设置信息。

    参数:
    logger: 日志记录器
    normalized_groups: 规范化后的图像组字典
    model_names: 使用的深度学习模型名称列表
    methods: 使用的传统图像处理方法列表
    threshold: 相似度阈值
    num_workers: 并行处理的工作进程数
    batch_size: 每批处理的图像组数量
    """
    total_images = sum(len(paths) for paths in normalized_groups.values())
    logger.info("开始图像相似度分析任务")
    logger.info(f"图像组总数: {len(normalized_groups)}")
    logger.info(f"图像总数: {total_images}")
    logger.info(f"使用的模型: {', '.join(model_names)}")
    logger.info(f"使用的设备: {DEVICE}")
    logger.info(f"使用的相似度方法: {', '.join(methods) if methods else '仅使用深度学习模型'}")
    logger.info(f"相似度阈值: {threshold}")
    logger.info(f"并行进程数: {num_workers}")
    logger.info(f"批处理大小: {batch_size}")

def main(image_groups: Dict[str, List[str]], threshold: float = Config.DEFAULT_THRESHOLD, num_workers: int = Config.DEFAULT_NUM_WORKERS, 
         model_names: List[str] = Config.DEFAULT_MODEL_NAMES, methods: List[str] = Config.DEFAULT_METHODS, 
         output_dir: str = '', log_file: str = 'similarity_log.txt', batch_size: int = Config.DEFAULT_BATCH_SIZE) -> List[Dict]:
    # 设置环境并获取日志记录器
    logger = setup_environment(output_dir, log_file)
    
    # 规范化图像组路径
    normalized_groups = normalize_image_groups(image_groups)
    
    # 记录分析设置信息
    log_analysis_settings(logger, normalized_groups, model_names, methods, threshold, num_workers, batch_size)
    
    # 开始处理图像组
    logger.info("开始处理图像组...")
    results = process_groups_in_batches(normalized_groups, threshold, num_workers, model_names, methods, logger, batch_size)
    
    # 处理完成，准备保存结果
    logger.info("处理完成，准备保存结果...")
    final_output_file = os.path.join(output_dir, 'image_similarity_results.xlsx')
    save_results_to_excel(results, final_output_file, logger)
    
    # 记录分析完成信息
    logger.info(f"分析完成，找到 {len(results)} 对相似图像")
    logger.info(f"结果已保存至: {final_output_file}")
    
    # 返回结果
    return results
    
if __name__ == "__main__":
    # 图片url表 
    img_df = pd.read_excel('img_info_20241010_1427.xlsx')
    # 首先，过滤出 wjfl 为 1300 （土壤剖面） 的图片
    filtered_df = img_df[img_df['wjfl'] == 1300]
    # 使用 groupby 和 agg 来创建字典
    img_url_dict = filtered_df.groupby('glbh')['wjlj'].agg(list).to_dict()
    groups = dict(list(img_url_dict.items())[:5])
    output_directory = '../result'
    os.makedirs(output_directory, exist_ok=True)
    log_file = os.path.join(output_directory, 'similarity_log.txt')
    # 指定要使用的传统方法 可选：histogram, hash, mse, template, phash，ssim，glcm
    methods = ['histogram', 'hash', 'mse', 'template', 'phash']
    # 深度学习预训练模型 可选：resnet50, vgg16, densenet121，inception_v3, mobilenet_v2, efficientnet_b0, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
    model_names = ['resnet50','vgg16','densenet121']
    # 相似度阈值
    threshold = 0.9
    # 并行进程数(CPU核数)
    num_workers = 48
    # 批次大小
    batch_size = 50
    # 设置多进程启动方法
    set_start_method()
    main(groups, threshold=threshold, num_workers=num_workers, model_names=model_names, 
         methods=methods, output_dir=output_directory, log_file=log_file, batch_size=batch_size)


