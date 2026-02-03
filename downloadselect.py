import os
import json
import argparse
import re
import subprocess
import sys
import urllib.request
import socket

# 默认 URL
BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'

def get_unique_scenes(jsonl_path):
    """从 jsonl 文件中提取唯一的 scene_id"""
    scenes = set()
    print(f"正在解析标注文件: {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print(f"错误: 文件不存在 {jsonl_path}")
        sys.exit(1)
        
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                video_path = data.get('video', '')
                match = re.search(r'(scene\d{4}_\d{2})', video_path)
                if match:
                    scenes.add(match.group(1))
    except Exception as e:
        print(f"错误: 无法读取 JSONL 文件: {e}")
        sys.exit(1)
    # 必须排序，确保两台机器看到的列表顺序是一致的，这样倒序才有意义
    return sorted(list(scenes))

def check_connection(url, proxy=None):
    """预检查 URL 是否可达"""
    print(f"正在预检连接: {url}")
    
    try:
        if proxy:
            os.environ['http_proxy'] = proxy
            os.environ['https_proxy'] = proxy
            
        req = urllib.request.Request(
            url, 
            method='HEAD', 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"❌ 连接检查失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ScanNet 双机协同下载工具')
    parser.add_argument('--jsonl', required=True, help='标注文件路径 (JSONL)')
    parser.add_argument('--out_dir', required=True, help='根目录')
    parser.add_argument('--v1', action='store_true', help='使用 v1 版本 (默认 v2)')
    parser.add_argument('--types', nargs='+', default=['.sens'], help='下载类型')
    parser.add_argument('--proxy', help='HTTP代理地址', default=None)
    # 新增倒序参数
    parser.add_argument('--reverse', action='store_true', help='开启倒序模式 (从最后一个文件向前下载)')
    args = parser.parse_args()

    # 1. 获取场景 ID
    scene_ids = get_unique_scenes(args.jsonl)
    if not scene_ids:
        print("未找到场景 ID。")
        return

    # 关键修改：如果是第二台机器，翻转列表
    if args.reverse:
        print(f"🔄 **已启用倒序模式**：将从 {scene_ids[-1]} 开始向前下载")
        scene_ids.reverse()
    else:
        print(f"▶️ **正序模式**：将从 {scene_ids[0]} 开始向后下载")

    release = 'v1/scans' if args.v1 else 'v2/scans'
    
    # 2. 验证第一个链接 (检查当前列表的第一个，无论是正序还是倒序)
    test_scene = scene_ids[0]
    test_file = f"{test_scene}{args.types[0]}"
    test_url = f"{BASE_URL}{release}/{test_scene}/{test_file}"
    
    if not check_connection(test_url, args.proxy):
        print("\n⚠️ 致命错误: 无法连接到 ScanNet 服务器。")
        return

    # 3. 生成下载列表
    aria2_input_path = 'aria2_input_rev.txt' if args.reverse else 'aria2_input.txt'
    wget_script_path = 'wget_download_rev.sh' if args.reverse else 'wget_download.sh'
    
    has_aria2 = subprocess.run(['which', 'aria2c'], capture_output=True).returncode == 0

    if has_aria2:
        with open(aria2_input_path, 'w') as f:
            for scene_id in scene_ids:
                for ext in args.types:
                    url = f"{BASE_URL}{release}/{scene_id}/{scene_id}{ext}"
                    local_dir = os.path.join(args.out_dir, scene_id)
                    f.write(f"{url}\n")
                    f.write(f"  dir={local_dir}\n")
                    f.write(f"  out={scene_id}{ext}\n")
                    f.write(f"  user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\n")
    else:
        with open(wget_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            for scene_id in scene_ids:
                local_dir = os.path.join(args.out_dir, scene_id)
                f.write(f"mkdir -p {local_dir}\n")
                for ext in args.types:
                    url = f"{BASE_URL}{release}/{scene_id}/{scene_id}{ext}"
                    proxy_cmd = f"-e use_proxy=yes -e http_proxy={args.proxy} -e https_proxy={args.proxy}" if args.proxy else ""
                    f.write(f"wget -c {proxy_cmd} --no-check-certificate --user-agent=\"Mozilla/5.0\" '{url}' -P '{local_dir}' -nc\n")

    print(f"\n准备下载 {len(scene_ids)} 个场景。")
    print("="*50)

    try:
        if has_aria2:
            print(f"🚀 启动 aria2c ({'倒序' if args.reverse else '正序'})...")
            cmd = [
                'aria2c', 
                '-i', aria2_input_path, 
                '--continue=true',
                '--max-connection-per-server=8', 
                '--split=8', 
                '--min-split-size=1M', 
                '--max-concurrent-downloads=5',
                '--check-certificate=false',
                '--connect-timeout=60',
                '--max-tries=5',
                '--retry-wait=5',
                '--file-allocation=none',
                '--summary-interval=10'
            ]
            if args.proxy:
                cmd.append(f'--all-proxy={args.proxy}')

            subprocess.run(cmd)
        else:
            print("⚠️ 未检测到 aria2c，使用 wget...")
            subprocess.run(['bash', wget_script_path])
    finally:
        pass

if __name__ == "__main__":
    main()