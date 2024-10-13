import os
import sys
import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from tqdm import tqdm as tqdm_sync
from collections import defaultdict
import logging
from datetime import datetime

# 平台特定的请求头
WINDOWS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
}

LINUX_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
}

# 根据操作系统选择适当的请求头
DEFAULT_HEADERS = WINDOWS_HEADERS if sys.platform.startswith('win') else LINUX_HEADERS

def setup_logger(log_path):
    logger = logging.getLogger('image_downloader')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

async def download_image(session, url, save_path, max_retries=3, headers=None):
    headers = headers or {}

    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=10, headers=headers) as response:
                response.raise_for_status()
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
            return True, None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"下载失败: {url}. 错误: {str(e)}. 重试中 ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(1)  # 在重试之前等待1秒
            else:
                logger.error(f"下载失败: {url}. 错误: {str(e)}. 已达到最大重试次数.")
                return False, f"{url}\t{str(e)}\n"
    return False, f"{url}\t未知错误\n"

async def download_batch(session, batch, base_path, column_names, max_retries, headers, failed_downloads):
    tasks = []
    counters = defaultdict(lambda: defaultdict(int))
    
    for _, row in batch.iterrows():
        glbh = row[column_names['glbh']]
        wjfl = row[column_names['wjfl']]
        url = row[column_names['url']]
        save_dir = os.path.join(base_path, wjfl, glbh)
        os.makedirs(save_dir, exist_ok=True)
        
        counters[glbh][wjfl] += 1
        file_name = f"{glbh}_{wjfl}_{counters[glbh][wjfl]}.jpg"
        save_path = os.path.join(save_dir, file_name)
        
        tasks.append(download_image(session, url, save_path, max_retries, headers))
    
    results = await tqdm.gather(*tasks, desc="下载进度")
    success_count = sum(result[0] for result in results)
    
    for result in results:
        if not result[0] and result[1]:
            failed_downloads.write(result[1])
    
    return success_count, len(tasks)

async def download_images(df, base_path, wjfl_to_download, column_names, max_retries=3, headers=None, batch_size=1000, failed_downloads_path=None):
    logger.info(f"DataFrame 形状: {df.shape}")
    logger.info(f"指定的 wjfl: {wjfl_to_download}")
    logger.info(f"DataFrame 中的 wjfl 唯一值: {df[column_names['wjfl']].unique()}")
    
    df[column_names['wjfl']] = df[column_names['wjfl']].astype(str)
    df[column_names['glbh']] = df[column_names['glbh']].astype(str)
    
    df_filtered = df[df[column_names['wjfl']].isin(map(str, wjfl_to_download))]
    logger.info(f"过滤后的 DataFrame 形状: {df_filtered.shape}")
    
    if df_filtered.empty:
        logger.warning("过滤后的 DataFrame 为空，请检查 wjfl_to_download 是否正确")
        return
    
    total_success = 0
    total_tasks = 0
    
    total_batches = (len(df_filtered) + batch_size - 1) // batch_size
    
    with open(failed_downloads_path, 'w', encoding='utf-8') as failed_downloads:
        async with aiohttp.ClientSession() as session:
            with tqdm_sync(total=total_batches, desc="总体进度", unit="batch") as pbar:
                for i in range(0, len(df_filtered), batch_size):
                    batch = df_filtered.iloc[i:i+batch_size]
                    success, tasks = await download_batch(session, batch, base_path, column_names, max_retries, headers, failed_downloads)
                    total_success += success
                    total_tasks += tasks
                    logger.info(f"批次 {i//batch_size + 1}/{total_batches} 完成: 成功 {success}/{tasks}")
                    pbar.update(1)
    
    logger.info(f"总共下载: {total_tasks}, 成功: {total_success}, 失败: {total_tasks - total_success}")

if __name__ == "__main__":
    
    # 用户url表格文件路径
    excel_path = "img.xlsx"
    
    # 用户指定的保存路径
    base_path = "../img_folder"
    
    # 用户指定的日志文件路径
    log_path = "../img_folder/download_log.txt"
    
    # 用户指定的下载失败记录文件路径
    failed_downloads_path = "../img_folder/failed_downloads.txt"
    
    # 用户指定的 wjfl 列表
    wjfl_to_download = [1300, 1303]
    
    column_names = {
        'wjfl': 'wjfl',
        'glbh': 'glbh',
        'url': 'wjlj'
    }
    max_retries = 5 # 最大重试次数

    batch_size = 50  # 每批处理的任务数
    
    # 设置日志
    logger = setup_logger(log_path)
    
    # 用户可以自定义请求头，如果不提供，将使用默认的平台特定请求头
    custom_headers = {
        'Referer': 'https://www.example.com',  # 可以根据实际情况修改
        'Cookie': 'your_cookie_here'  # 如果需要的话，可以添加Cookie
    }
    
    # 合并自定义请求头和默认请求头
    # headers = {**DEFAULT_HEADERS, **custom_headers}
    headers = None
    # 请求头
    logger.info(f"请求头: {DEFAULT_HEADERS}")
    df = pd.read_excel(excel_path)
    logger.info(f"加载的 DataFrame 形状: {df.shape}")
    logger.info(f"DataFrame 列名: {df.columns}")
    
    asyncio.run(download_images(df, base_path, wjfl_to_download, column_names, max_retries, headers, batch_size, failed_downloads_path))
