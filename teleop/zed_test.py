import pyzed.sl as sl

# Create a Camera object
zed = sl.Camera()

# Create and set initialization parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution
init_params.camera_fps = 30  # Set FPS

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error opening the camera: {err}")
    exit(1)

print("Camera opened successfully.")

import os

def rename_files_in_directory(directory):
    # 获取文件夹中的所有文件
    files = os.listdir(directory)
    
    # 按照文件名排序（可选）
    files.sort()
    
    # 遍历并重命名文件
    for i, filename in enumerate(files):
        # 获取文件扩展名
        file_extension = os.path.splitext(filename)[1]
        # 新文件名，数字编号从1开始
        new_filename = f"{i + 1}{file_extension}"
        # 构造完整路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")

# 使用方法
directory_path = "your/directory/path"
rename_files_in_directory(directory_path)


