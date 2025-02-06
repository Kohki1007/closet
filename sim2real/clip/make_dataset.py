import time
import torch
import cv2
import numpy as np
import random
import os

from torchvision import transforms
from PIL import Image

dataset_current_2 = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/dataset/current_normal.pt")
dataset_target_2 = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/dataset/target_normal.pt")
dataset_current_3 = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset/dataset_real/dataset/current_norm.pt")
dataset_target_3 = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset/dataset_real/dataset/target_norm.pt")

dataset_current_2 = dataset_current_2.view(-1, 2, 256, 256)
dataset_current_2 = dataset_current_2[:, 0].unsqueeze(1)
dataset_current_2 = dataset_current_2.expand(-1, 3, -1, -1)
dataset_target_2 = dataset_target_2.view(-1, 2, 256, 256)
# attension = dataset_target_2[:, 1]
# dataset_target_2 = dataset_target_2[:, 0].unsqueeze(1)
# dataset_target_2 = dataset_target_2.expand(-1, 3, -1, -1)

# dataset_target_2 = torch.cat(())

ramdom_position = random.sample(range(0, 5), 3)

dataset_current_3 = dataset_current_3[:, ramdom_position].view(-1, 2, 256, 256)
dataset_current_3 = dataset_current_3[:, 0].unsqueeze(1)
dataset_current_3 = dataset_current_3.expand(-1, 3, -1, -1)
dataset_target_3 = dataset_target_3[:, ramdom_position].view(-1, 2, 256, 256)
# dataset_target_3 = dataset_target_3[:, 0].unsqueeze(1)
# dataset_target_3 = dataset_target_3.expand(-1, 3, -1, -1)

current_image = torch.cat((dataset_current_2, dataset_current_3), dim = 0)
target_image = torch.cat((dataset_target_2, dataset_target_3), dim = 0)

def r_ints_nodup(a, b, k):
            ns = []
            while len(ns) < k:
                n = np.random.randint(a, b)
                if not n in ns:
                    ns.append(n)
            nss = np.sort(np.array(ns))
            return nss


nums = np.array(range(13200))
test_idx = r_ints_nodup(0, 13200, 2000)
train_idx = np.delete(nums, test_idx, 0)

test_idx = test_idx.tolist()
train_idx = train_idx.tolist()

current_image_train = current_image[train_idx]
target_image_train = target_image[train_idx]
current_image_test = current_image[test_idx]
target_image_test = target_image[test_idx]

torch.save(current_image_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/current_train_RGB_2.pt")
torch.save(current_image_test, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/current_test_RGB_2.pt")
torch.save(target_image_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/target_train_2.pt")
torch.save(target_image_test, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/target_test_2.pt")

# dataset_depth = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/dataset_base/current_base.pt")
# # attension = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset/dataset/current_target.pt")
# target_depth = torch.load("/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/dataset_base/target_base.pt")

# print(dataset_depth.shape)
# print(torch.max(dataset_depth))

# ramdom_position = random.sample(range(0, 15), 3)
# ramdom_position = [1, 5, 13]
# dataset_depth = dataset_depth[:, ramdom_position]
# target_depth = target_depth[:, ramdom_position]



# def make_attension(left, size, position, step):
#         transform = transforms.Resize((256, 256))
#         image_left = np.copy(left)
#         save_attension = image_left.copy()
#         attension = np.zeros([1, 256, 256])
#         anno_h_left = []
#         anno_w_left = []
#         # print(image_left.shape)
#         for h in range (image_left.shape[0]):
#             for j in range (image_left.shape[1]):

#                 if np.all(image_left[h, j, :] == [0, 0, 0]):
#                     image_left[h, j, :] = [1, 2, 3]
#                 elif np.all(image_left[h, j, :] > [240, 240, 240]):
#                     image_left[h, j, :] = [1, 2, 3]
#                 elif np.all(image_left[h, j, :] == [1, 1, 1]):
#                     image_left[h, j, :] = [1, 2, 3]
#                 elif np.all(image_left[h, j, :] < [30, 30, 30]):
#                     image_left[h, j, :] = [1, 2, 3]

#                 if image_left[h, j, 0] == image_left[h, j, 1] and image_left[h, j, 1] == image_left[h, j, 2]:
#                     anno_h_left.append(h)
#                     anno_w_left.append(j)
#                     # save_attension[h, j] = [255, 255, 255]
#                 # else:
#                 #     save_attension[h, j] = [0, 0, 0]

#         min_h = min(anno_h_left)
#         max_h = max(anno_h_left)
#         min_w = min(anno_w_left)
#         max_w = max(anno_w_left)

#         attension[:, min_h:max_h, min_w:max_w] = 1

#         attension = torch.from_numpy(attension.astype(np.float32)).clone()

#         return attension

# for size in range(dataset_depth.shape[0]):
#     print(f"size = {size}")
#     for position in range(len(ramdom_position)):
#         id = ramdom_position[position]
#         print(f"position = {position}")
#         for i in range(41):
#             image = cv2.imread(f"/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/image/{size}_{id}_{i}.png")
#             if i == 40:
#                 target_depth[size, position, :, 1] = make_attension(image, size, position, 40)
#             else:
#                 image = cv2.imread(f"/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/image/{size}_{id}_{i}.png")
#                 dataset_depth[size, position, i, 1] = make_attension(image, size, position, i)

#             target_depth[:, :, :, 0] /= torch.max(target_depth[:, :, :, 0])
#             dataset_depth[:, :, :, 0] /= torch.max(dataset_depth[:, :, :, 0])


# torch.save(target_depth, "/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/dataset/target_normal.pt")
# torch.save(dataset_depth, "/home/engawa/py_ws/visual_servo/src/network/dataset_Imager_normalize/target_image_generator/dataset/current_normal.pt")
# target_depth = []
# dataset_depth = []