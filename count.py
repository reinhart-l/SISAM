import os

def count_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    count = 0
    for root, dirs, files in os.walk(folder_path):
        count += sum(1 for file in files if os.path.splitext(file)[-1].lower() in image_extensions)
    return count

def main():
    base_folder = '/home3/yuchu/MedSAM/data/SIS_private/'  # 请替换为你的应用路径
    images_folder = os.path.join(base_folder, 'images')
    masks_folder = os.path.join(base_folder, 'annotations')

    images_count = count_images_in_folder(images_folder)
    masks_count = count_images_in_folder(masks_folder)

    print(f"Number of images in 'images' folder: {images_count}")
    print(f"Number of images in 'masks' folder: {masks_count}")

if __name__ == "__main__":
    main()
