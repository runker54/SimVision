import os
import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from collections import defaultdict

# 默认的浏览器请求头
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
}

async def download_image(session, url, save_path, max_retries=3, headers=None):
    headers = headers or {}

    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=10, headers=headers) as response:
                response.raise_for_status()
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"下载失败: {url}. 错误: {str(e)}. 重试中 ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(1)  # 在重试之前等待1秒
            else:
                print(f"下载失败: {url}. 错误: {str(e)}. 已达到最大重试次数.")
    return False

async def download_images(df, base_path, wjfl_to_download, column_names, max_retries=3, headers=None):
    print(f"DataFrame 形状: {df.shape}")
    print(f"指定的 wjfl: {wjfl_to_download}")
    print(f"DataFrame 中的 wjfl 唯一值: {df[column_names['wjfl']].unique()}")
    
    # 确保 wjfl 和 glbh 列为字符串类型
    df[column_names['wjfl']] = df[column_names['wjfl']].astype(str)
    df[column_names['glbh']] = df[column_names['glbh']].astype(str)
    
    # 过滤指定的wjfl
    df_filtered = df[df[column_names['wjfl']].isin(map(str, wjfl_to_download))]
    print(f"过滤后的 DataFrame 形状: {df_filtered.shape}")
    
    if df_filtered.empty:
        print("过滤后的 DataFrame 为空，请检查 wjfl_to_download 是否正确")
        return
    
    # 创建文件夹结构并初始化计数器
    counters = defaultdict(lambda: defaultdict(int))
    for wjfl in map(str, wjfl_to_download):
        for glbh in df_filtered[df_filtered[column_names['wjfl']] == wjfl][column_names['glbh']].unique():
            folder_path = os.path.join(base_path, wjfl, glbh)
            os.makedirs(folder_path, exist_ok=True)
            print(f"创建文件夹: {folder_path}")
    
    # 准备下载任务
    tasks = []
    async with aiohttp.ClientSession() as session:
        for _, row in df_filtered.iterrows():
            glbh = row[column_names['glbh']]
            wjfl = row[column_names['wjfl']]
            url = row[column_names['url']]
            save_dir = os.path.join(base_path, wjfl, glbh)
            
            # 使用计数器为每个glbh和wjfl组合生成唯一的文件名
            counters[glbh][wjfl] += 1
            file_name = f"{glbh}_{wjfl}_{counters[glbh][wjfl]}.jpg"
            save_path = os.path.join(save_dir, file_name)
            
            tasks.append(download_image(session, url, save_path, max_retries, headers))
        
        print(f"准备下载的任务数: {len(tasks)}")
        
        # 使用异步方式下载
        results = await tqdm.gather(*tasks, desc="下载进度")
    
    print(f"总共下载: {len(tasks)}, 成功: {sum(results)}, 失败: {len(tasks) - sum(results)}")

# 使用示例
if __name__ == "__main__":
    # 用户指定参数
    excel_path = r"img.xlsx"
    base_path = r"img_folder"
    wjfl_to_download = [1300, 1302]  # 指定需要下载的wjfl
    column_names = {
        'wjfl': 'wjfl',
        'glbh': 'glbh',
        'url': 'wjlj'
    }
    max_retries = 3 # 重试次数
    
    # 用户可以自定义请求头，如果不提供，将使用默认的请求头
    custom_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.example.com',  # 可以根据实际情况修改
        'Cookie': 'your_cookie_here'  # 如果需要的话，可以添加Cookie
    }
    
    # 如果想使用默认请求头，可以注释掉下面这行，或者设置 headers = None
    # headers = {**DEFAULT_HEADERS, **custom_headers}
    headers = None

    # 加载 DataFrame
    df = pd.read_excel(excel_path)
    print(f"加载的 DataFrame 形状: {df.shape}")
    print(f"DataFrame 列名: {df.columns}")
    
    asyncio.run(download_images(df, base_path, wjfl_to_download, column_names, max_retries, headers))
