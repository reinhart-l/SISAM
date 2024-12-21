import os
import json
from PIL import Image,ImageDraw
import shutil
import numpy as np
#------------------------------------------------绘制binary_mask---------------------------------------
# class_to_id = {
#     "class1": 1,
#     "class2": 2,
#     "class3": 3,
#     "class4": 4,
#     "class5": 5,
#     "class6": 6,
#     "class7": 7,
#     "class8": 8,
#     "class9": 9,
#     "class10": 10,
#     "class11": 11,
# }
#
# def get_last_file_number(target_folder):
#     files = [f for f in os.listdir(target_folder) if f.endswith('.png')]
#
#     if not files:
#         return 0
#
#     files.sort()
#     last_file = files[-1]
#     last_number = int(last_file.split('_')[0].split('.png')[0])  # 提取文件名中的数字部分
#     return last_number + 1
# def create_semantic_masks(json_file, output_mask_base_path):
#     # 读取 JSON 文件
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#
#     # 获取图像尺寸
#     image_width = data['imageWidth']
#     image_height = data['imageHeight']
#
#     # 创建一个全局的语义分割 mask
#     global_mask = np.zeros((image_height, image_width), dtype=np.uint8)
#
#     # 为每个类别创建一个单独的二值 mask
#     masks = {class_id: np.zeros((image_height, image_width), dtype=np.uint8) for class_id in class_to_id.values()}
#
#     # 遍历所有的标注形状
#     for shape in data['shapes']:
#         label = shape['label']
#         points = shape['points']
#
#         # 获取对应的 class id
#         class_id = class_to_id.get(label, 0)  # 如果没有匹配，默认为 0 (背景)
#
#         # 将 points 转换为 tuple 列表
#         polygon = [(int(point[0]), int(point[1])) for point in points]
#
#         # 使用 PIL 的 ImageDraw 在 global_mask 和对应类别 mask 上绘制多边形
#         img_global = Image.fromarray(global_mask)
#         draw_global = ImageDraw.Draw(img_global)
#         draw_global.polygon(polygon, outline=class_id, fill=class_id)
#         global_mask = np.array(img_global)
#
#         # 更新对应类别的二值 mask：前景 255，背景 0
#         img_class = Image.fromarray(masks[class_id])
#         draw_class = ImageDraw.Draw(img_class)
#         draw_class.polygon(polygon, outline=255, fill=255)  # 前景区域填充 255
#         masks[class_id] = np.array(img_class)
#
#     # # 保存全局 mask 图片
#     # global_mask_img = Image.fromarray(global_mask)
#     # global_mask_output_path = f"{output_mask_base_path}.png"
#     # global_mask_img.save(global_mask_output_path)
#     # print(f"Global semantic mask saved to {global_mask_output_path}")
#
#     # 保存每个类别的二值 mask 图片
#     for class_id, mask in masks.items():
#         if np.any(mask):  # 如果 mask 中有非零值，说明有该类别
#             mask_img = Image.fromarray(mask)
#             mask_output_path = f"{output_mask_base_path}_class{class_id}.png"
#             mask_img.save(mask_output_path)
#             print(f"Binary mask for class {class_id} saved to {mask_output_path}")
# img_path="./data/surgery/images/"
# json_path="./data/surgery/json/"
# target_img_path="./data/SIS/train_private/images/"
# target_mask_path="./data/SIS/train_private/annotations/"
#
# cnt=get_last_file_number(target_img_path)
#
# for root, dirs, files in os.walk(img_path):
#     if len(dirs) == 0:
#         # 按文件名中的数字顺序进行排序
#         files = sorted(files, key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x)
#         for file in files:
#             name = file.split('.')[0]
#             json_file = os.path.join(json_path, name + '.json')
#             if not os.path.exists(json_file):
#                 continue
#             if file.endswith(".jpg"):
#                 png_file_name = f"{cnt:06d}.png"
#                 png_path = os.path.join(target_img_path, png_file_name)
#
#                 # 打开 JPG 文件并转换为 PNG
#                 with Image.open(os.path.join(root, file)) as img:
#                     img.save(png_path, 'PNG')
#                     print(f"Converted {os.path.join(root, file)} to {png_path}")
#
#                 new_name = png_file_name.split('.')[0]
#                 create_semantic_masks(json_file, os.path.join(target_mask_path, new_name))
#                 cnt += 1

#-----------------------------------------生成image2mask的json文件-------------------------------------
img_path="./data/SIS/train_private/images/"
mask_path="./data/SIS/train_private/annotations/"
mapping = {}
for root, dirs, files in os.walk(mask_path):
    print(root, dirs, files)
    if len(dirs) == 0:
        for file in files:
            img_name = file.split('_class')[0] + '.png'
            seq_num=root.split('/')[-1]
            mapping[root+'/'+file]=img_path+seq_num+'/'+img_name
print(mapping)
# 将字典转换为 JSON 格式并写入文件
with open("./data/SIS/train_private/label2image.json", "w") as json_file:
    json.dump(mapping, json_file, indent=4)


img_path="./data/SIS/train_private/images/"
mask_path="./data/SIS/train_private/annotations/"
image_masks = {}
for root, dirs, files in os.walk(img_path):
    if len(dirs) == 0:
        for file in files:
            img_name = file.split('.')[0]
            image_path = os.path.join(root, file)
            # List all mask files corresponding to the image
            seq_num = root.split('/')[-1]
            masks = [f for f in os.listdir(mask_path+seq_num+'/') if img_name in f]
            mask_paths = [os.path.join(mask_path+seq_num+'/', mask) for mask in masks]
            # Add to dictionary
            if mask_paths:
                image_masks[image_path] = mask_paths
print(image_masks)
# 将字典转换为 JSON 格式并写入文件
with open("./data/SIS/train_private/image2label.json", "w") as json_file:
    json.dump(image_masks, json_file, indent=4)