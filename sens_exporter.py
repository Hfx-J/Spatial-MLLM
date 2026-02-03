import os
import struct
import numpy as np
import zlib
import imageio
import cv2
import png
import argparse
import sys
import concurrent.futures
import time
import subprocess
import glob
from pathlib import Path
from datetime import datetime

# 压缩类型定义
COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
             return zlib.decompress(self.depth_data)
        raise ValueError(f"Unknown depth compression type: {compression_type}")

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
             return imageio.imread(self.color_data)
        raise ValueError(f"Unknown color compression type: {compression_type}")

class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen))).decode('utf-8')
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, frame_skip=1):
        if not os.path.exists(output_path): os.makedirs(output_path)
        for f in range(0, len(self.frames), frame_skip):
            out_file = os.path.join(output_path, f"{f}.png")
            if os.path.exists(out_file): continue
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            with open(out_file, 'wb') as file:
                writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16, greyscale=True)
                writer.write(file, depth.tolist())

    def export_color_images(self, output_path, frame_skip=1):
        if not os.path.exists(output_path): os.makedirs(output_path)
        for f in range(0, len(self.frames), frame_skip):
            out_file = os.path.join(output_path, f"{f}.jpg")
            if os.path.exists(out_file): continue
            color = self.frames[f].decompress_color(self.color_compression_type)
            imageio.imwrite(out_file, color)

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path): os.makedirs(output_path)
        for f in range(0, len(self.frames), frame_skip):
            out_file = os.path.join(output_path, f"{f}.txt")
            if os.path.exists(out_file): continue
            np.savetxt(out_file, self.frames[f].camera_to_world, fmt='%f')

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path): os.makedirs(output_path)
        np.savetxt(os.path.join(output_path, 'intrinsic_color.txt'), self.intrinsic_color, fmt='%f')
        np.savetxt(os.path.join(output_path, 'intrinsic_depth.txt'), self.intrinsic_depth, fmt='%f')

    def make_video_ffmpeg(self, color_dir, video_dir, scene_name, fps=30):
        if not os.path.exists(video_dir): os.makedirs(video_dir)
        output_v_path = os.path.join(video_dir, f"{scene_name}.mp4")

        if os.path.exists(output_v_path):
            return f"⏭️  视频已存在，跳过"
        
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-framerate', str(fps),
            '-i', os.path.join(color_dir, '%d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            output_v_path
        ]
        try:
            subprocess.run(cmd, check=True)
            return f"🎬 视频生成成功"
        except subprocess.CalledProcessError:
            return f"❌ FFmpeg 执行失败"

def process_single_file(args):
    file_path, frame_skip = args
    filename = os.path.basename(file_path)
    scene_name = os.path.splitext(filename)[0]
    base_dir = os.path.dirname(file_path)
    
    try:
        sd = SensorData(file_path)
        color_path = os.path.join(base_dir, 'color')
        video_path = os.path.join(base_dir, 'video')
        
        sd.export_color_images(color_path, frame_skip=frame_skip)
        sd.export_depth_images(os.path.join(base_dir, 'depth'), frame_skip=frame_skip)
        sd.export_poses(os.path.join(base_dir, 'pose'), frame_skip=frame_skip)
        sd.export_intrinsics(os.path.join(base_dir, 'intrinsic'))
        
        video_msg = sd.make_video_ffmpeg(color_path, video_path, scene_name)
        return f"✅ {scene_name} 处理完成 | {video_msg}"
    except Exception as e:
        return f"❌ {scene_name} 出错: {str(e)}"

def get_all_sens_files(root_dir):
    """获取所有 .sens 文件"""
    search_path = os.path.join(root_dir, "scene*", "*.sens")
    sens_files = glob.glob(search_path)
    
    if not sens_files:
        sens_files = glob.glob(os.path.join(root_dir, "*.sens"))
    
    return set(sens_files)

def get_processed_marker_path(sens_file):
    """为每个 .sens 文件生成一个处理标记文件路径"""
    base_dir = os.path.dirname(sens_file)
    return os.path.join(base_dir, '.processed_marker')

def is_file_processed(sens_file):
    """检查文件是否已被处理"""
    marker_path = get_processed_marker_path(sens_file)
    if not os.path.exists(marker_path):
        return False
    
    # 读取标记文件中的时间戳，比较是否比 .sens 文件新
    try:
        with open(marker_path, 'r') as f:
            marker_time = float(f.read().strip())
        sens_mtime = os.path.getmtime(sens_file)
        return marker_time >= sens_mtime
    except:
        return False

def mark_file_processed(sens_file):
    """标记文件已处理"""
    marker_path = get_processed_marker_path(sens_file)
    with open(marker_path, 'w') as f:
        f.write(str(time.time()))

def process_new_files(root_dir, frame_skip=1, num_workers=4):
    """处理新发现的文件"""
    all_sens_files = get_all_sens_files(root_dir)
    new_files = [f for f in all_sens_files if not is_file_processed(f)]
    
    if not new_files:
        return 0
    
    tasks = [(f, frame_skip) for f in sorted(new_files)]
    print(f"\n📦 发现 {len(tasks)} 个新文件需要处理")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            # 从结果中提取文件路径并标记已处理
            for task in tasks:
                scene_name = os.path.splitext(os.path.basename(task[0]))[0]
                if scene_name in result and "✅" in result:
                    mark_file_processed(task[0])
    
    return len(tasks)

def watch_directory(root_dir, frame_skip=1, num_workers=4, check_interval=10):
    """
    持续监控目录，发现新文件就处理
    
    Args:
        root_dir: 监控的根目录
        frame_skip: 帧跳过参数
        num_workers: 并行工作进程数
        check_interval: 检查间隔（秒）
    """
    print(f"🔍 开始监控目录: {root_dir}")
    print(f"⚙️  检查间隔: {check_interval}秒 | 并行数: {num_workers} | 帧跳过: {frame_skip}")
    print(f"{'='*60}")
    
    # 首次运行时处理所有现有文件
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 初始扫描...")
    count = process_new_files(root_dir, frame_skip, num_workers)
    if count > 0:
        print(f"✅ 初始处理完成，共处理 {count} 个文件\n")
    else:
        print(f"📭 暂无新文件需要处理\n")
    
    # 进入监控循环
    try:
        while True:
            time.sleep(check_interval)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] 🔄 检查新文件...", end='')
            
            count = process_new_files(root_dir, frame_skip, num_workers)
            
            if count > 0:
                print(f" ✅ 处理了 {count} 个新文件")
            else:
                print(f" 📭 无新文件")
                
    except KeyboardInterrupt:
        print(f"\n\n⏹️  监控已停止")
        sys.exit(0)

def process_sens_directory_parallel(root_dir, frame_skip=1, num_workers=4):
    """一次性批量处理（原始功能）"""
    sens_files = get_all_sens_files(root_dir)
    tasks = [(f, frame_skip) for f in sorted(sens_files)]
    
    print(f"📂 搜寻结束。共发现 {len(tasks)} 个 .sens 文件。")
    if not tasks: return

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ScanNet .sens 文件处理工具（支持监控模式）'
    )
    parser.add_argument('--input_dir', required=True, help='ScanNet 根目录')
    parser.add_argument('--frame_skip', type=int, default=1, help='帧跳过间隔')
    parser.add_argument('--num_workers', type=int, default=2, help='并行工作进程数')
    parser.add_argument('--watch', action='store_true', help='启用监控模式，持续检测新文件')
    parser.add_argument('--check_interval', type=int, default=10, 
                        help='监控模式下的检查间隔（秒），默认10秒')
    
    args = parser.parse_args()

    if args.watch:
        # 监控模式
        watch_directory(
            args.input_dir, 
            args.frame_skip, 
            args.num_workers,
            args.check_interval
        )
    else:
        # 一次性批量处理模式
        process_sens_directory_parallel(
            args.input_dir, 
            args.frame_skip, 
            args.num_workers
        )