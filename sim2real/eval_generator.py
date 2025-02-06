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
parser.add_argument("--weight_path", type=str, default="/home/engawa/py_ws/visual_servo/src/network/sim2real/log/generator_many_1214/weight/weights_last.pth", help="重みのパス")
parser.add_argument("--save_path", type=str, default="generator_real", help="保存先のパス")
parser.add_argument("--current_path", type=str, default="dataset/dataset_real/goal_image/dataset/input_many.pth", help="入力１") ####### image1
parser.add_argument("--target_path", type=str, default="dataset/target_test_1203.pt", help="入力１")
parser.add_argument("--action_path", type=str, default=None, help="アクション")

args = parser.parse_args()


train_base = train_function(args.model, args.weight_path, args.epoch, args.lrt, args.batch)
train_base.make_dir(args.save_path)

num_elements = 3  # リストの要素数
min_value = 0      # 最小値
max_value = 1499   # 最大値
random_list = [random.randint(min_value, max_value) for _ in range(num_elements)]


current = torch.load("dataset/eval/generator_accuracy/current.pt")
target = torch.load("dataset/eval/generator_accuracy/target.pt")

for i in range(180):
    for j in range(10):
        input = current[j*30 : (j+1)*30, i]
        output = target[j*30 : (j+1)*30, i]

        train_base.eval_generator(input, output, i)

train_base.draw_accuracy_list()


