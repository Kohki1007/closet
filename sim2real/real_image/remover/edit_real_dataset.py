import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

base_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/real_image"
transform = transforms.Resize((256, 256))

# image_path = f"/home/engawa/py_ws/visual_servo/src/network/dataset/real_depth_image/08.png"

# image = Image.open(image_path)
# image = np.array(image)
# print(image)
image_list = []

for i in range(26):
    if i != 1:
        image_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/real_image/remover/depth_image_resize_transform_{i}.png"

        image = Image.open(image_path)
        image = np.array(image)
        image = torch.from_numpy(image.astype(np.float32)).clone()
        # print(image.shape)
        image /= torch.max(image)
        # image = torch.where(image == 0, 1, image)
        image = image.unsqueeze(0)
        image = transform(image)
        image = image.squeeze()
        image_list.append(image.clone())

image_dataset = torch.stack(image_list)
print(image_dataset.shape)
torch.save(image_dataset, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_real/remover/input.pt")