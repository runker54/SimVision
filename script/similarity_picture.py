import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor
from urllib.request import urlretrieve
from urllib.parse import urlparse
import imagehash
from PIL import Image
from typing import Dict, List, Tuple, Any
import logging

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

def load_image(path, as_gray=True):
    """
    加载图像,支持本地文件和URL
    """
    if is_url(path):
        temp_file, _ = urlretrieve(path)
        img = cv2.imread(temp_file)
        os.unlink(temp_file)
    else:
        img = cv2.imread(path)
    
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
    
    # 根据需要加载灰度和彩色图像
    if 'ssim' in methods or 'histogram' in methods or 'mse' in methods:
        img1_gray = load_image(path1)
        img2_gray = load_image(path2)
        img1_color = load_image(path1, as_gray=False)
        img2_color = load_image(path2, as_gray=False)
    
    # 计算各种相似度
    if 'ssim' in methods:
        results['ssim'] = ssim_similarity(img1_gray, img2_gray)
    
    if 'histogram' in methods:
        results['histogram'] = histogram_similarity(img1_color, img2_color)
    
    if 'mse' in methods:
        results['mse'] = mse_similarity(img1_gray, img2_gray)
    
    if 'phash' in methods:
        results['phash'] = phash_similarity(path1, path2)
    
    # 处理深度学习方法
    deep_learning_methods = [m for m in methods if m in ['resnet50', 'vgg16', 'inceptionv3']]
    for method in deep_learning_methods:
        similarity_func = get_deep_learning_similarity(method)
        results[method] = similarity_func(path1, path2)
    
    # 计算综合相似度
    combined_sim = sum(results.values()) / len(results) if results else 0
    return combined_sim, results

def process_group(group, methods, threshold):
    results = []
    n = len(group)
    for i in range(n):
        for j in range(i+1, n):
            try:
                similarity, details = compare_images(group[i], group[j], methods)
                if similarity > threshold:
                    results.append((group[i], group[j], similarity, details))
            except Exception as e:
                print(f"Error processing images {group[i]} and {group[j]}: {e}")
    return results

def find_similar_images(groups, methods, threshold, num_processes, logger):
    logger.info(f"开始查找相似图片,使用方法: {methods}, 阈值: {threshold}, 进程数: {num_processes}")
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_results = list(executor.map(process_group, groups.values(), [methods] * len(groups), [threshold] * len(groups)))
    
    for group_name, group_result in zip(groups.keys(), all_results):
        if group_result:
            results[group_name] = group_result
            logger.info(f"组 {group_name} 找到 {len(group_result)} 对相似图片")
        else:
            results[group_name] = [(None, None, 0, {})]
            logger.info(f"组 {group_name} 没有找到相似图片")

    return results

def save_results_to_excel(results: Dict[str, List[Tuple[str, str, float, Dict[str, float]]]], methods: List[str], filename: str = 'similar_images.xlsx', logger=None):
    data = []
    max_similar_images = 0

    for group_name, similarities in results.items():
        if not similarities or similarities[0][0] is None:
            row = [group_name, "无相似性图片", ""] + [""] * len(methods)
            data.append(row)
        else:
            similarity_groups = {}
            for sim in similarities:
                key = frozenset([sim[0], sim[1]])
                similarity_groups.setdefault(key, []).append(sim)

            for group in similarity_groups.values():
                avg_similarity = sum(sim[2] for sim in group) / len(group)
                similar_images = list(set().union(*[{sim[0], sim[1]} for sim in group]))
                max_similar_images = max(max_similar_images, len(similar_images))
                
                row = [group_name, "有相似性图片", avg_similarity]
                row.extend([sum(sim[3].get(m, 0) for sim in group) / len(group) for m in methods])
                row.extend(similar_images)
                data.append(row)

    # 确保所有行有相同的长度
    max_row_length = max(len(row) for row in data)
    columns = ['组名', '相似性', '平均相似度'] + methods + [f'相似图片{i+1}' for i in range(max_row_length - 3 - len(methods))]

    for row in data:
        row.extend([''] * (max_row_length - len(row)))

    df = pd.DataFrame(data, columns=columns)
    df.to_excel(filename, index=False)
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
    logger.info(f"图片组: {groups}")
    logger.info(f"使用方法: {methods}")
    logger.info(f"相似度阈值: {threshold}")
    logger.info(f"并行进程数: {num_processes}")
    
    # 查找相似图片
    similar_pairs = find_similar_images(groups, methods, threshold, num_processes, logger)
    
    # 保存结果到Excel
    save_results_to_excel(similar_pairs, methods, 'similar_images.xlsx', logger)
    
    # 输出结果日志
    for group_name, similarities in similar_pairs.items():
        logger.info(f"组名: {group_name}")
        if not similarities or similarities[0][0] is None:
            logger.info("无相似性图片")
        else:
            similarity_groups = {}
            for sim in similarities:
                key = frozenset([sim[0], sim[1]])
                similarity_groups.setdefault(key, []).append(sim)
            
            for i, group in enumerate(similarity_groups.values(), 1):
                similar_images = list(set().union(*[{s[0], s[1]} for s in group]))
                logger.info(f"相似图片组 {i}:")
                logger.info(f"相似图片: {', '.join(similar_images)}")
                avg_similarity = sum(s[2] for s in group) / len(group)
                logger.info(f"平均相似度: {avg_similarity:.2f}")
                for method in methods:
                    avg_method_similarity = sum(s[3].get(method, 0) for s in group) / len(group)
                    logger.info(f"{method}: {avg_method_similarity:.2f}")
                logger.info("---")
    
    logger.info("图片相似度分析完成")
    return similar_pairs

# 示例用法
if __name__ == "__main__":
    groups = {
        "组1": [
                'https://sanpu.iarrp.cn/ssp-dccy/2023-11-10/520330/70da4515-f196-4ee4-867d-2f055614b030.jpg',
                "https://sanpu.iarrp.cn/ssp-dccy/2023-11-10/520330/70da4515-f196-4ee4-867d-2f055614b031.jpg",
                "https://sanpu.iarrp.cn/ssp-dccy/2023-11-10/520330/70da4515-f196-4ee4-867d-2f055614b032.jpg",
        ],
        "组2": [
            r"C:\Users\Runker\Desktop\C.jpg",
            r"C:\Users\Runker\Desktop\D.jpg",
            r"C:\Users\Runker\Desktop\E.jpg",
            r"C:\Users\Runker\Desktop\F.jpg",
            r"C:\Users\Runker\Desktop\G.jpg",
            r"C:\Users\Runker\Desktop\H.jpg",
        ],
    }
    methods = ['ssim', 'histogram', 'phash']
    threshold = 0.8
    num_processes = 2
    results = main(groups, methods, threshold, num_processes)