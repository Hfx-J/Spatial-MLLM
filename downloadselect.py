import os
import json
import argparse
import urllib.request
import tempfile
import re

BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'
# 加入 .mp4 到文件类型列表中
FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply', '_2d-label.zip', '.mp4']

def get_unique_scenes(jsonl_path):
    """从 jsonl 文件中提取唯一的 scene_id"""
    scenes = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            video_path = data.get('video', '')
            # 使用正则匹配 sceneXXXX_XX
            match = re.search(r'scene\d{4}_\d{2}', video_path)
            if match:
                scenes.add(match.group())
    return sorted(list(scenes))

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    if os.path.isfile(out_file):
        print(f"  [跳过] 已存在: {os.path.basename(out_file)}")
        return

    print(f"  [下载] {url} -> {out_file}")
    try:
        # 使用临时文件下载，防止中断导致文件损坏
        fd, tmp_path = tempfile.mkstemp(dir=out_dir)
        os.close(fd)
        urllib.request.urlretrieve(url, tmp_path)
        os.rename(tmp_path, out_file)
    except Exception as e:
        print(f"  [错误] 下载失败 {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description='根据标注文件选择性下载 ScanNet 场景。')
    parser.add_argument('--jsonl', required=True, help='标注文件路径 (JSONL)')
    parser.add_argument('--out_dir', required=True, help='下载保存目录')
    parser.add_argument('--v1', action='store_true', help='是否使用 v1 版本')
    parser.add_argument('--types', nargs='+', default=['.mp4', '.sens'], help='需要下载的文件后缀')
    args = parser.parse_args()

    release = 'v1/scans' if args.v1 else 'v2/scans'
    
    # 1. 解析场景 ID
    print(f"正在从 {args.jsonl} 提取场景 ID...")
    scene_ids = get_unique_scenes(args.jsonl)
    print(f"找到 {len(scene_ids)} 个唯一场景: {scene_ids}")

    # 2. 确认条款
    print("\n继续操作即表示您同意 ScanNet 的使用条款。")
    print("按回车键开始下载，或按 Ctrl+C 退出。")
    input("")

    # 3. 循环下载
    for scene_id in scene_ids:
        print(f"\n正在处理场景: {scene_id}")
        # 注意：ScanNet 的 .mp4 通常放在特定的 tasks 目录下或者由 .sens 转换
        # 如果官方服务器直接支持 .mp4 路径如下：
        for ext in args.types:
            url = f"{BASE_URL}{release}/{scene_id}/{scene_id}{ext}"
            out_path = os.path.join(args.out_dir, scene_id, f"{scene_id}{ext}")
            download_file(url, out_path)

if __name__ == "__main__":
    main()