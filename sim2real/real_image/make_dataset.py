import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

base_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/real_image"
transform = transforms.Resize((256, 256))

# image_path = f"/home/engawa/py_ws/visual_servo/src/network/dataset/real_depth_image/08.png"

# image = Image.open(image_path)
# image = np.array(image)
# print(image)
image_list = []

for i in range(23):
    if i != 8 and i != 16:
        number = f"{i}".zfill(2)
        path = f"/{number}.png"
        image_path = f"/home/engawa/py_ws/visual_servo/src/network/dataset/real_depth_image/{number}.png"

        image = Image.open(image_path)
        image = np.array(image)
        image = torch.from_numpy(image.astype(np.float32)).clone()
        # print(image.shape)
        image /= torch.max(image)
        image = torch.where(image == 0, 1, image)
        image = image.unsqueeze(0)
        image = transform(image)
        image = image.squeeze()
        image_list.append(image.clone())
        # image = image.to('cpu').detach().numpy().copy()

        # depth = Image.fromarray(image)
        # depth.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/real_image/{number}.png")

image_dataset = torch.stack(image_list)
print(image_dataset.shape)
torch.save(image_dataset, "src/network/sim2real/dataset/real_input.pt")
