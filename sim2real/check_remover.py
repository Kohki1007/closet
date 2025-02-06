import torch
import sys 
import argparse
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
import random
from PIL import Image

from train_function import train_function

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='goal_image_generator', help="学習するモデル名")
parser.add_argument("--lrt", type=str, default=0.00001, help="学習率")
parser.add_argument("--epoch", type=str, default=300000000000, help="エポック数")
parser.add_argument("--batch", type=str, default=32, help="バッチサイズ")
parser.add_argument("--weight_path", type=str, default="log/remover_sim_ur_3/weight/weights_last.pth", help="重みのパス")
parser.add_argument("--save_path", type=str, default="generator_real", help="保存先のパス")
parser.add_argument("--current_path", type=str, default="dataset/dataset_remover_sim/current_train.pt", help="入力１") ####### image1
parser.add_argument("--target_path", type=str, default="dataset/dataset_remover_sim/target_train.pt", help="入力１")
parser.add_argument("--action_path", type=str, default=None, help="アクション")

args = parser.parse_args()


train_base = train_function(args.model, args.weight_path, args.epoch, args.lrt, args.batch)
train_base.make_dir(args.save_path)

# num_elements = 3  # リストの要素数
# min_value = 0      # 最小値
# max_value = 1499   # 最大値
# random_list = [random.randint(min_value, max_value) for _ in range(num_elements)]


test_input = torch.load(args.current_path)
test_output = torch.load(args.target_path)

# print(test_dataset.shape)

# depth = test_dataset[0, 3].to('cpu').detach().numpy().copy()
# depth *= 255
# depth = Image.fromarray((depth).astype(np.uint8))
# # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
# depth.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_real/goal_image/dataset/test.png")

# test_dataset = test_dataset[:, 3].unsqueeze(1)

# test_dataset = torch.load(args.current_path)[:5]
# test_dataset = test_dataset.view(-1, 1, 256, 256)
# test_dataset = test_dataset[random_list]

# # print(test_dataset.shape)
# test_dataset_target = torch.load(args.target_path)[:5]
# test_dataset_target = test_dataset_target.view(-1, 1, 256, 256)
# test_dataset_target = test_dataset_target[random_list]

for i in range(1000):
    train_base.forward_solo(test_input[i * 28:(i+1)*28], test_output[i * 28:(i+1)*28], i)

