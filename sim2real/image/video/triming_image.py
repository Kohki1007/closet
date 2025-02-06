import numpy
import cv2
import os

# def load_image(image_path):
#     image = cv2.imread(image_path)
#     # transform = transforms.Compose([
#     #     transforms.Toimage()
#     # ])
#     # image_image = transform(image)
#     return image_image
directory_list = [1, 2, 3, 4, 5]

for i in directory_list:
    directory = f"{i}/"
    prefix = "env" 

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory, filename)
            try:
                image = cv2.imread(image_path)
                # print(image.shape)
                image = image[:, 400:1150]
                # image = torch.where(image == 0, 1, image)
                # images.append(image)
                print(os.path.join(f"triming_{i}", filename))
                cv2.imwrite( os.path.join(f"triming_{i}", filename), image)
                # print(f"Loaded: {filename} -> image Shape: {image.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    # images = torch.stack(images)

    # image = 