import os
import shutil

# 设置文件路径
txt_file = 'refined_datasets/rubble/refined_image_list.txt'  # txt文件路径
source_folder = 'datasets/rubble/images/train/rgbs'  # 包含.pt文件的源文件夹路径
destination_folder = 'refined_datasets/rubble/train/rgbs'  # 目标文件夹路径

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 读取txt文件中的每一行
with open(txt_file, 'r') as file:
    for line in file:
        # 获取.jpg文件名，然后更改扩展名为.pt
        jpg_filename = line.strip()
        pt_filename = jpg_filename.replace('.jpg', '.jpg')
        
        # 构建源文件和目标文件的完整路径
        source_file = os.path.join(source_folder, pt_filename)
        destination_file = os.path.join(destination_folder, pt_filename)
        
        # 如果源文件存在，则复制到目标文件夹
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")
        else:
            print(f"File not found: {source_file}")