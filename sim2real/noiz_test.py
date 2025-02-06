import torch
from PIL import Image
import numpy as np
import cv2

from train_function import train_function

train_base = train_function()
train_image = torch.load("dataset/dataset_remover/current_train.pt")
for i in range(train_image.shape[0]):
    image = train_image[i, 0]
    train_image[i, 0] = train_base.add_noise_shapes(image).clone()

torch.save(train_image, "dataset/dataset_remover/noised_current_train_UR.pt")
train_image = []
test_image = torch.load("dataset/dataset_remover/current_val.pt")
for i in range(test_image.shape[0]):
    image = test_image[i, 0]
    test_image[i, 0] = train_base.add_noise_shapes(image).clone()

torch.save(test_image, "dataset/dataset_remover/noised_current_val_UR.pt")

# image = torch.load("dataset/dataset_remover/test_image_noiz.pt")
# sample_image = image[0]
# std = 0.1

# noized_image = train_base.add_noise_shapes(sample_image)
# save_image = noized_image.to('cpu').detach().numpy().copy()

# save_image *= 255
# depth = Image.fromarray((save_image).astype(np.uint8))
# depth.save(f"image/test_noiz/noiz_all.png")


