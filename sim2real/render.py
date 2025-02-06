from render_function import rendar
import torch
import sys 
import argparse
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms


# parser = argparse.ArgumentParser()
# parser.add_argument("--env", type=str, default="object", help="urdf の種類")
# parser.add_argument("--num_envs", type=int, default=200, help="環境数")
# parser.add_argument("--position_path", type=str, default='goal_image_generator', help="position のパス")
# parser.add_argument("--rot_path", type=str, default=0.0003, help="rotation のパス")
# parser.add_argument("--props_path", type=str, default=0.0003, help="rotation のパス")
# parser.add_argument("--image_size", type=int, default=224, help="画像サイズ")
# parser.add_argument("--repeat", type=int, default=0, help="繰り返し数")
# parser.add_argument("--dataset_name", type=str, help="ディレクトリ名")
# parser.add_argument("--dataset_path", type=str, default="1210", help="ファイル名")

# args = parser.parse_args()

env = "goal"
num_envs = 200
position_path = "remover_sim/closet_posistion.pt"
rot_path = "remover_sim/closet_rot.pt"
props_path = "remover_sim/closet_size.pt"
image_size = 256
repeat = 0
dataset_name = "remover_sim"
dataset_path = "1210"





render = rendar(env, position_path, rot_path, props_path, image_size, num_envs, repeat)
env_number, position_number, step_number = render.parameter()
for position in range(position_number):
    render.set_closet_position(position)
    for step in range(step_number):
        render.pre_physics_step(step)
        render.get_image(step, position, dataset_name, dataset_path)
