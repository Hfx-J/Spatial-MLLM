#!/usr/bin/env python3
"""
简单版本：移动 scannet 视频文件
"""

import os
import shutil
from pathlib import Path


def move_videos_simple(scannet_root="/mnt/huangfeixuan/scannet"):
    """简单版本的视频移动脚本"""
    scannet_path = Path(scannet_root)
    target_dir = scannet_path / "video"
    
    # 创建目标目录
    target_dir.mkdir(exist_ok=True)
    
    # 遍历所有 scene 目录
    moved = 0
    for scene_dir in scannet_path.glob("scene*"):
        if not scene_dir.is_dir():
            continue
            
        video_file = scene_dir / "video" / f"{scene_dir.name}.mp4"
        
        if video_file.exists():
            target_file = target_dir / video_file.name
            
            if not target_file.exists():
                print(f"移动: {video_file.name}")
                shutil.move(str(video_file), str(target_file))
                moved += 1
            else:
                print(f"跳过: {video_file.name} (已存在)")
    
    print(f"\n完成！共移动 {moved} 个文件")


if __name__ == "__main__":
    move_videos_simple("/mnt/huangfeixuan/scannet")