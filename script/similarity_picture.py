import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlparse
from urllib.request import urlretrieve
from typing import Dict, List, Tuple
import tempfile
import imagehash
from PIL import Image
import logging
from tqdm import tqdm
import gc

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

def load_image(path, as_gray=True, max_retries=5, backoff_factor=0.5):
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
            img = cv2.imread(temp_file_path)
            os.unlink(temp_file_path)
        except requests.RequestException as e:
            logging.error(f"加载URL图像 {path} 时发生网络错误: {e}")
            return None
        except Exception as e:
            logging.error(f"无法加载URL图像 {path}: {e}")
            return None
    else:
        img = cv2.imread(path)
    
    if img is None:
        logging.error(f"无法加载图像 {path}")
        return None

    if as_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# 非深度学习相似度方法
def ssim_similarity(img1, img2):
    """
    计算结构相似性指数(SSIM)
    """
    return ssim(img1, img2)

def histogram_similarity(img1, img2):
    """
    计算直方图相似度
    """
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def mse_similarity(img1, img2):
    """
    计算均方误差(MSE)相似度
    """
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return 1 - (err / 255**2)  # 归一化并反转MSE以获得相似度

def phash_similarity(path1, path2):
    """
    计算感知哈希(pHash)相似度
    """
    img1 = Image.open(path1) if not is_url(path1) else Image.open(urlretrieve(path1)[0])
    img2 = Image.open(path2) if not is_url(path2) else Image.open(urlretrieve(path2)[0])
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return 1 - (hash1 - hash2) / 64  # 归一化到[0, 1]区间

# 深度学习方法 (仅在需要时导入和初始化)
def get_deep_learning_similarity(method):
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
    from sklearn.metrics.pairwise import cosine_similarity

    def load_model(model_name):
        if model_name == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False, pooling='avg'), resnet_preprocess, (224, 224)
        elif model_name == 'vgg16':
            return VGG16(weights='imagenet', include_top=False, pooling='avg'), vgg_preprocess, (224, 224)
        elif model_name == 'inceptionv3':
            return InceptionV3(weights='imagenet', include_top=False, pooling='avg'), inception_preprocess, (299, 299)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def extract_features(path, model, preprocess_func, target_size):
        img = load_image_for_deep_learning(path, target_size)
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_func(img_array)
        features = model.predict(img_array)
        return features.flatten()

    def load_image_for_deep_learning(path, target_size):
        if is_url(path):
            temp_file, _ = urlretrieve(path)
            img = Image.open(temp_file).convert('RGB')
            os.unlink(temp_file)
        else:
            img = Image.open(path).convert('RGB')
        img = img.resize(target_size)
        return image.img_to_array(img)

    model, preprocess_func, target_size = load_model(method)

    def compute_similarity(path1, path2):
        features1 = extract_features(path1, model, preprocess_func, target_size)
        features2 = extract_features(path2, model, preprocess_func, target_size)
        return cosine_similarity([features1], [features2])[0][0]

    return compute_similarity

# 主比较函数
def compare_images(path1, path2, methods):
    """
    比较两张图片的相似度,使用多种方法
    """
    results = {}
    
    img1 = load_image(path1)
    img2 = load_image(path2)
    
    if img1 is None or img2 is None:
        return 0, results

    if 'ssim' in methods:
        results['ssim'] = ssim(img1, img2)
    
    if 'phash' in methods:
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        hash1 = imagehash.phash(img1_pil)
        hash2 = imagehash.phash(img2_pil)
        results['phash'] = 1 - (hash1 - hash2) / 64  # 归一化到[0, 1]区间
    
    # 显式删除大型对象
    del img1, img2, img1_pil, img2_pil
    
    combined_sim = sum(results.values()) / len(results) if results else 0
    return combined_sim, results

def process_group(group_name, group_images, methods, threshold, logger):
    logger.info(f"开始处理组 {group_name}")
    results = []
    n = len(group_images)
    for i in tqdm(range(n), desc=f"处理组 {group_name}"):
        for j in range(i+1, n):
            try:
                similarity, details = compare_images(group_images[i], group_images[j], methods)
                if similarity > threshold:
                    results.append((group_images[i], group_images[j], similarity, details))
            except Exception as e:
                logger.error(f"处理图片 {group_images[i]} 和 {group_images[j]} 时出错: {e}")
    
    if results:
        logger.info(f"组 {group_name} 找到 {len(results)} 对相似图片")
    else:
        logger.info(f"组 {group_name} 没有找到相似图片")
    
    return group_name, results

def process_groups_in_batches(groups, methods, threshold, num_processes, logger, batch_size=50):
    all_results = {}
    group_items = list(groups.items())
    total_batches = (len(group_items) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(group_items), batch_size), total=total_batches, desc="处理批次"):
        batch = dict(group_items[i:i+batch_size])
        logger.info(f"处理批次 {i//batch_size + 1}/{total_batches}, 包含 {len(batch)} 组")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_group = {executor.submit(process_group, name, images, methods, threshold, logger): name 
                               for name, images in batch.items()}
            
            for future in concurrent.futures.as_completed(future_to_group):
                group_name, group_result = future.result()
                all_results[group_name] = group_result

    return all_results

def save_results_to_excel(results: Dict[str, List[Tuple[str, str, float, Dict[str, float]]]], methods: List[str], filename: str = 'similar_images.xlsx', logger=None):
    data = []
    for group_name, similarities in results.items():
        if not similarities:
            row = [str(group_name), "无相似性图片", "", ""] + [""] * len(methods)
        else:
            similar_images = set()
            for sim in similarities:
                similar_images.add(sim[0])
                similar_images.add(sim[1])
            
            avg_similarity = sum(sim[2] for sim in similarities) / len(similarities)
            method_scores = {m: sum(sim[3].get(m, 0) for sim in similarities) / len(similarities) for m in methods}
            
            row = [
                str(group_name),
                "有相似性图片",
                f"{len(similarities)}对",
                f"{len(similar_images)}张"
            ]
            row.extend([f"{avg_similarity:.4f}"] + [f"{method_scores.get(m, 0):.4f}" for m in methods])
            row.append(", ".join(similar_images))
        
        data.append(row)

    columns = ['组名', '相似性', '相似对数', '相似图片数', '平均相似度'] + methods + ['相似图片']
    df = pd.DataFrame(data, columns=columns)
    
    # 指定组名列为字符串类型
    df['组名'] = df['组名'].astype(str)
    
    # 对结果进行排序
    df = df.sort_values(by='平均相似度', ascending=False)
    
    # 使用 ExcelWriter 来设置列格式
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # 设置组名列为文本格式
        for cell in worksheet['A']:
            cell.number_format = '@'

    if logger:
        logger.info(f"结果已保存到 {filename}")
    else:
        print(f"结果已保存到 {filename}")

# 添加日志配置
def setup_logger(log_file='image_similarity.log'):
    logger = logging.getLogger('image_similarity')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# 主函数,适用于Jupyter Notebook
def main(groups, methods, threshold, num_processes, log_file='image_similarity.log'):
    """
    主函数,执行图片相似度分析并输出结果
    """
    logger = setup_logger(log_file)
    logger.info("开始执行图片相似度分析")
    logger.info(f"图片组: 共{len(groups)}组")
    logger.info(f"使用方法: {methods}")
    logger.info(f"相似度阈值: {threshold}")
    logger.info(f"并行进程数: {num_processes}")
    
    # 查找相似图片
    similar_pairs = process_groups_in_batches(groups, methods, threshold, num_processes, logger)
    
    # 保存结果到Excel
    save_results_to_excel(similar_pairs, methods, 'similar_images.xlsx', logger)
    
    logger.info("图片相似度分析完成")
    return similar_pairs

# 示例用法
if __name__ == "__main__":
    # 图片url表 
    img_df = pd.read_excel('../table/img_info_20241010_1427.xlsx')
    # 首先，过滤出 wjfl 为 1300 的图片
    filtered_df = img_df[img_df['wjfl'] == 1300]
    # 使用 groupby 和 agg 来创建字典
    img_url_dict = filtered_df.groupby('glbh')['wjlj'].agg(list).to_dict()
    # 使用main函数进行图片相似度计算
    # groups = dict(list(img_url_dict.items()))   # 图片组 这里只示例前5组
    groups = img_url_dict
    # methods = ['ssim','phash','vgg16','resnet50','inceptionv3']  # 相似度计算方法  这里选择ssim,histogram,phash三种方法组合。 # resnet50 vgg16 inceptionv3
    methods = ['phash','ssim','histogram']  # 相似度计算方法  这里选择ssim,histogram,phash三种方法组合。 # resnet50 vgg16 inceptionv3
    threshold = 0.6  # 相似度阈值
    num_processes = min(os.cpu_count(), 48)  # 减少进程数以降低磁盘压力
    main(groups,methods,threshold,num_processes)
