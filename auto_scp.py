import os
import time
import subprocess
import argparse
import sys

def check_and_upload(local_root, remote_user, remote_host, remote_root, file_types):
    """
    遍历目录，寻找已下载完成但未上传的文件
    """
    # 遍历本地目录
    for root, dirs, files in os.walk(local_root):
        for file in files:
            # 1. 筛选目标后缀 (例如 .sens, .mp4)
            if not any(file.endswith(ext) for ext in file_types):
                continue
            
            # 忽略我们自己生成的标记文件 (.done)
            if file.endswith('.done'):
                continue

            file_path = os.path.join(root, file)
            aria2_file = file_path + '.aria2'
            done_marker = file_path + '.done'

            # 2. 核心判断逻辑
            # - 原文件存在
            # - .aria2 文件不存在 (说明 aria2 已经下载完并删除了控制文件)
            # - .done 标记文件不存在 (说明还没上传过)
            if os.path.exists(file_path) and \
               not os.path.exists(aria2_file) and \
               not os.path.exists(done_marker):
                
                # 计算相对路径，保持目录结构
                # 例如 local: .../scannet/scene0000_00/test.sens
                # rel_path: scene0000_00/test.sens
                rel_path = os.path.relpath(file_path, local_root)
                
                # 远程目标文件的完整路径
                remote_file_path = os.path.join(remote_root, rel_path)
                # 远程目标目录
                remote_dir = os.path.dirname(remote_file_path)

                print(f"\n[发现新文件] {file}")
                print(f"   -> 正在上传至: {remote_host}:{remote_file_path}")

                try:
                    # 3. 步骤A: 确保远程目录存在 (使用 ssh)
                    # scp 无法自动创建多级远程目录，所以必须先 ssh mkdir
                    subprocess.run(
                        ['ssh', f'{remote_user}@{remote_host}', f'mkdir -p {remote_dir}'],
                        check=True
                    )

                    # 4. 步骤B: 执行 SCP 上传
                    # -B: 批处理模式 (防止询问密码)
                    # -q: 安静模式
                    subprocess.run(
                        ['rsync', '-avz', '--partial', '--progress', file_path, f'{remote_user}@{remote_host}:{remote_file_path}'],
                        check=True
                    )

                    print(f"   ✅ 上传成功！")
                    
                    # 5. 步骤C: 创建本地标记文件
                    # 创建一个空的 .done 文件，下次循环直接跳过
                    with open(done_marker, 'w') as f:
                        f.write('uploaded')
                        
                    # 可选：如果你硬盘空间不够，上传成功后可以删除本地源文件
                    # os.remove(file_path) 
                    # print("   🗑️ 本地文件已删除以释放空间")

                except subprocess.CalledProcessError as e:
                    print(f"   ❌ 上传失败: {e}")
                except Exception as e:
                    print(f"   ❌ 发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='自动监控并上传已下载的 ScanNet 文件')
    parser.add_argument('--local_dir', required=True, help='本地监控的根目录 (下载目录)')
    parser.add_argument('--remote_user', required=True, help='远程主机用户名 (例如 huangfeixuan)')
    parser.add_argument('--remote_host', required=True, help='远程主机IP (例如 192.168.1.100)')
    parser.add_argument('--remote_dir', required=True, help='远程主机的存放根目录')
    parser.add_argument('--interval', type=int, default=60, help='检测间隔秒数 (默认 60秒)')
    
    args = parser.parse_args()

    print("="*50)
    print(f"📡 自动传输服务已启动")
    print(f"📂 监控本地: {args.local_dir}")
    print(f"🚀 目标主机: {args.remote_user}@{args.remote_host}")
    print(f"📂 目标路径: {args.remote_dir}")
    print("="*50)

    try:
        while True:
            # 扫描 .sens 和 .mp4
            check_and_upload(
                args.local_dir, 
                args.remote_user, 
                args.remote_host, 
                args.remote_dir, 
                ['.sens', '.mp4']
            )
            
            print(f"\r⏳ 扫描完成，休眠 {args.interval} 秒...", end="", flush=True)
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")

if __name__ == "__main__":
    main()