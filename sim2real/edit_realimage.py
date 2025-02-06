import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as T
import math
import random
from torchvision import transforms


def load_image_as_tensor(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    return image_tensor

def load_image(image_path):
    image = cv2.imread(image_path)

    # cv2.imshow("Grayscale Image", image)
    # cv2.waitKey(0)

    # print(image.shape)
    return image

def delete_edgenoise(image):
    save_image = image[0].clone()
    # print(save_image.shape)
    for j in range(save_image.shape[0]):
        for k in range(save_image.shape[1]):
            if k > 1 and k < 224: 
                right = save_image[j, k + 1]
                left = save_image[j, k - 1]
                # print(right)
                if right == 1 and left == 1:
                    save_image[j, k] = 1.
    
    return save_image 

########## 目標画像の生成
# for i in range(9):
#     i += 85
#     image_path = f"dataset/dataset_remover_real/output_edited/{i}.png"
#     image_np = load_image(image_path)
#     image_tensor = load_image_as_tensor(image_path)
#     image_tensor = delete_edgenoise(image_tensor)
#     image_tensor = image_tensor.squeeze().to('cpu').detach().numpy().copy()

#     save_image = image_tensor
#     # save_image[:, :58] = 1.
#     save_image[:, 120:132] = 1.
#     save_image[:, 187:] = 1.

#     # door_right = save_image[:, 232:]
#     # door_right = torch.where(door_right)

#     save_image *= 255
#     depth = Image.fromarray((save_image).astype(np.uint8))
#     depth.save(f"dataset/dataset_remover_real/output_filter/{i}.png")

# ##### dataset にまとめる
# current_image_tensor = []
# target_image_tensor = []
# for i in range(94):
#     current_path = f"dataset/dataset_remover_real/input/{i}.png"
#     target_path = f"dataset/dataset_remover_real/output_filter/{i}.png"
#     current_image_tensor.append(load_image_as_tensor(current_path))
#     target_image_tensor.append(load_image_as_tensor(target_path))

#     # save_image *= 255
#     # depth = Image.fromarray((save_image).astype(np.uint8))
#     # depth.save(f"dataset/dataset_remover_real/output_filter/{i}.png")
# current_image = torch.stack(current_image_tensor)
# target_image = torch.stack(target_image_tensor)
# print(current_image.shape)

# torch.save(current_image, "dataset/dataset_remover/real_input_UR.pt")
# torch.save(target_image, "dataset/dataset_remover/real_output_UR.pt")

transform = transforms.Resize((224, 224))

###### real と sim の統
real_input = torch.load("dataset/dataset_remover/real_input_UR.pt")
real_output = torch.load("dataset/dataset_remover/real_output_UR.pt")

real_input = transform(real_input)
real_ioutput = transform(real_output)
sim_input_train = torch.load("dataset/dataset_remover/noised_current_train_UR.pt")
sim_output_train = torch.load("dataset/dataset_remover/target_train.pt")

train_id = sorted([random.randint(0, 21999) for _ in range(1000)])

train_input = torch.cat((sim_input_train[train_id], real_input[:-10]))
train_output = torch.cat((sim_output_train[train_id], real_output[:-10]))

torch.save(train_input, "dataset/dataset_remover/mix_input_train.pt")
torch.save(train_output, "dataset/dataset_remover/mix_output_train.pt")

sim_input_test = torch.load("dataset/dataset_remover/noised_current_val_UR.pt")
sim_output_test = torch.load("dataset/dataset_remover/target_val.pt")

test_id = sorted([random.randint(0, 1999) for _ in range(100)])
test_input = torch.cat((sim_input_test[test_id], real_input[-10:]))
test_output = torch.cat((sim_output_test[test_id], real_output[-10:]))

torch.save(test_input, "dataset/dataset_remover/mix_input_test.pt")
torch.save(test_output, "dataset/dataset_remover/mix_output_test.pt")
# 実行
# tensors = load_images_with_prefix(directory, prefix)

# torch.save(tensors, "dataset/dataset_remover_real/real_input.pt")

