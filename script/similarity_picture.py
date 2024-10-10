import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlparse
import tempfile
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import cv2
from scipy.spatial.distance import cosine
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import hamming
import sys

# 设置日志记录
def setup_logger(log_file_path):
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])
    return logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 工具函数
def is_url(path):
    """
    检查给定的路径是否为URL
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def load_image(path, max_retries=5, backoff_factor=0.5):
    """
    加载图像,支持本地文件和URL，使用流式处理和临时文件
    """
    session = requests.Session()
    retry = Retry(total=max_retries, backoff_factor=backoff_factor)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    if is_url(path):
        try:
            response = session.get(path, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir()) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            image = Image.open(temp_file_path).convert('RGB')
            os.unlink(temp_file_path)
        except requests.RequestException as e:
            logging.error(f"加载URL图像 {path} 时发生网络错误: {e}")
            return None
        except Exception as e:
            logging.error(f"无法加载URL图像 {path}: {e}")
            return None
    else:
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            logging.error(f"无法加载本地图像 {path}: {e}")
            return None
    
    return image

# Siamese 网络特征提取
class SiameseFeatureExtractor:
    def __init__(self, model_name='resnet50', logger=None):
        self.model_name = model_name
        self.model = self._get_model()
        self.model = self.model.to(device)
        self.model.eval()
        self.logger = logger

    def _get_model(self):
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif self.model_name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
        
        # 去掉最后的全连接层
        if self.model_name.startswith('resnet'):
            model.fc = torch.nn.Identity()
        elif self.model_name.startswith('vgg'):
            model.classifier = torch.nn.Identity()
        elif self.model_name.startswith('densenet'):
            model.classifier = torch.nn.Identity()
        
        return model

    def extract_features(self, img_path):
        try:
            image = load_image(img_path)
            if image is None:
                return None
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.model(image)
            return features.cpu().numpy().flatten()
        except Exception as e:
            if self.logger:
                self.logger.error(f"无法提取图像特征 {img_path}: {e}")
            return None

# 计算余弦相似度
def compute_cosine_similarity(feat1, feat2):
    return cosine_similarity([feat1], [feat2])[0][0]

def calculate_histogram_similarity(img1_path, img2_path):
    """
    计算两张图片的直方图相似度
    """
    # 使用 os.path.normpath 来规范化路径
    img1_path = os.path.normpath(img1_path)
    img2_path = os.path.normpath(img2_path)
    
    # 尝试使用 PIL 加载图像
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
    except Exception as e:
        print(f"无法打开图像文件: {e}")
        return None
    
    # 将 PIL 图像转换为 OpenCV 格式
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    # 计算直方图
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    return 1 - cosine(hist1, hist2)  # 转换为相似度

def calculate_ssim(img1_path, img2_path):
    """
    计算两张图片的结构相似性指数（SSIM）
    """
    # 使用 os.path.normpath 来规范化路径
    img1_path = os.path.normpath(img1_path)
    img2_path = os.path.normpath(img2_path)
    
    # 尝试使用 PIL 加载图像
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
    except Exception as e:
        print(f"无法打开图像文件: {e}")
        return None
    
    # 将 PIL 图像转换为 OpenCV 格式
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    # 转换为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 确保两张图片大小相同
    height, width = img1_gray.shape
    img2_gray = cv2.resize(img2_gray, (width, height))
    
    # 计算 SSIM
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
    return ssim_value

def calculate_average_hash(img_path):
    """
    计算图像的平均哈希值
    """
    try:
        # 使用 PIL 打开图像
        img = Image.open(img_path).convert('L')
        img = np.array(img)
    except Exception as e:
        print(f"无法打开图像文件 {img_path}: {e}")
        return None
    
    # 调整图像大小为8x8
    img = cv2.resize(img, (8, 8))
    
    # 计算平均值
    mean = np.mean(img)
    
    # 根据平均值生成哈希
    hash_value = 0
    for i in range(8):
        for j in range(8):
            if img[i, j] > mean:
                hash_value |= 1 << (i * 8 + j)
    
    return hash_value

def calculate_hash_similarity(img1_path, img2_path):
    """
    计算两张图片的平均哈希相似度
    """
    hash1 = calculate_average_hash(img1_path)
    hash2 = calculate_average_hash(img2_path)
    
    if hash1 is None or hash2 is None:
        return None
    
    # 计算汉明距离
    hamming_distance = bin(hash1 ^ hash2).count('1')
    
    # 转换为相似度（0-1范围）
    similarity = 1 - hamming_distance / 64.0
    
    return similarity

def calculate_mse(img1_path, img2_path):
    """计算均方误差（MSE）"""
    img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img1 is None or img2 is None:
        return None
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def calculate_psnr(img1_path, img2_path):
    """计算峰值信噪比（PSNR）"""
    mse = calculate_mse(img1_path, img2_path)
    if mse is None or mse == 0:
        return None
    return 20 * np.log10(255.0) - 10 * np.log10(mse)

def calculate_template_matching(img1_path, img2_path):
    """使用模板匹配计算相似度"""
    img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return None
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    return np.max(res)

# 处理单个图片组，返回所有相似的图片对及其相似度
def process_group(group_name, image_paths, threshold=0.8, model_name='resnet50', methods=None, logger=None):
    feature_extractor = SiameseFeatureExtractor(model_name, logger)
    features = []
    valid_image_paths = []
    for path in image_paths:
        path = os.path.normpath(path)
        feat = feature_extractor.extract_features(path)
        if feat is not None:
            features.append(feat)
            valid_image_paths.append(path)
    
    similar_pairs = []
    n = len(features)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = compute_cosine_similarity(features[i], features[j])
            result = {
                'Group Name': group_name,
                'Image 1': valid_image_paths[i],
                'Image 2': valid_image_paths[j],
                'Model Similarity': similarity
            }
            
            if methods:
                if 'histogram' in methods:
                    result['Histogram Similarity'] = calculate_histogram_similarity(valid_image_paths[i], valid_image_paths[j])
                if 'ssim' in methods:
                    result['SSIM Similarity'] = calculate_ssim(valid_image_paths[i], valid_image_paths[j])
                if 'hash' in methods:
                    result['Hash Similarity'] = calculate_hash_similarity(valid_image_paths[i], valid_image_paths[j])
                if 'mse' in methods:
                    result['MSE'] = calculate_mse(valid_image_paths[i], valid_image_paths[j])
                if 'psnr' in methods:
                    result['PSNR'] = calculate_psnr(valid_image_paths[i], valid_image_paths[j])
                if 'template' in methods:
                    result['Template Matching'] = calculate_template_matching(valid_image_paths[i], valid_image_paths[j])
            
            # 检查是否有任何相似度超过阈值
            if any(v >= threshold for k, v in result.items() if k != 'Group Name' and k != 'Image 1' and k != 'Image 2' and v is not None):
                similar_pairs.append(result)
    
    if logger:
        logger.info(f"组 {group_name} 处理完成，找到 {len(similar_pairs)} 对相似图片")
    return similar_pairs

# 修改 process_groups 函数
def process_groups(groups: Dict[str, List[str]], threshold=0.8, num_workers=8, model_name='resnet50', methods=None, logger=None):
    results = []
    with torch.no_grad():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_group = {executor.submit(process_group, group_name, image_paths, threshold, model_name, methods, logger): group_name 
                               for group_name, image_paths in groups.items()}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_group), total=len(groups), desc="处理图片组"):
                group_name = future_to_group[future]
                group_result = future.result()
                results.extend(group_result)
                
                if logger:
                    logger.info(f"组 {group_name} 处理完成")
                    if group_result:
                        logger.info(f"组 {group_name} 找到 {len(group_result)} 对相似图片")
                    else:
                        logger.info(f"组 {group_name} 没有找到相似的图片对")
    
    return results

# 保存结果到Excel
def save_results_to_excel(results: List[Dict[str, any]], filename, logger=None):
    if not results:
        if logger:
            logger.info("没有找到任何相似的图片对。")
        return
    
    df = pd.DataFrame(results)
    # 字符串格式保存组名编码列
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df.to_excel(filename, index=False, engine='openpyxl')
    if logger:
        logger.info(f"结果已保存到 {filename}")

# 添加这个函数来处理路径编码
def encode_path(path):
    if sys.platform.startswith('win'):
        return path.encode('gbk').decode('gbk')
    return path

# 修改 main 函数中的路径处理
def main(image_groups: Dict[str, List[str]], threshold=0.8, num_workers=8, model_name='resnet50', 
         methods=None, output_dir='', log_file='similarity_log.txt'):
    logger = setup_logger(os.path.join(output_dir, log_file))
    
    # 规范化所有图像路径并进行编码
    normalized_groups = {
        group_name: [encode_path(os.path.normpath(path)) for path in paths]
        for group_name, paths in image_groups.items()
    }
    
    logger.info(f"开始处理图片组，使用模型: {model_name}")
    logger.info(f"使用的相似度方法: {', '.join(methods) if methods else '仅深度学习模型'}")
    
    results = process_groups(normalized_groups, threshold, num_workers, model_name, methods, logger)
    
    final_output_file = os.path.join(output_dir, 'image_similarity_results.xlsx')
    save_results_to_excel(results, final_output_file, logger)
    return results

if __name__ == "__main__":
     # 图片url表 
    img_df = pd.read_excel('../table/img_info_20241010_1427.xlsx')
    # 首先，过滤出 wjfl 为 1300 的图片
    filtered_df = img_df[img_df['wjfl'] == 1300]
    # 使用 groupby 和 agg 来创建字典
    img_url_dict = filtered_df.groupby('glbh')['wjlj'].agg(list).to_dict()
    groups = img_url_dict
    output_directory = encode_path(r'C:\Users\Runker\Desktop\similarity_results')
    os.makedirs(output_directory, exist_ok=True)
    log_file = os.path.join(output_directory, 'similarity_log.txt')
    # 指定要使用的传统方法
    methods = ['histogram', 'ssim', 'hash', 'mse', 'psnr', 'template']
    # 指定要使用的深度学习模型
    model_name='resnet50'
    threshold = 0.8  # 相似度阈值
    num_processes = min(os.cpu_count(), 48)  # 减少进程数以降低磁盘压力
    # 深度学习选用resnet50
    results = main(image_groups, threshold=threshold, num_workers=num_processes, model_name=model_name, methods=methods, output_dir=output_directory, log_file=log_file)

