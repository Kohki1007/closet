import torch
from model import goal_image_generator, manipulator_remover
import numpy as np

input = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_sim/current.pt")
input = input.view(32000, 1, 224, 224)
target = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_remover_sim/current.pt")
target = target.view(32000, 1, 224, 224)
# action = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/action.pt")

goal = goal_image_generator()
goal.eval()
weight = torch.load("log/generator_many_1214/weight/weights_last.pth", map_location='cuda')
goal.load_state_dict(weight)
goal = goal.to("cuda")

remover = manipulator_remover()
remover.eval()
remover_weight = torch.load("log/remover_sim_ur_3/weight/weights_last.pth", map_location='cuda')
remover.load_state_dict(remover_weight)
remover = remover.to("cuda")

target_predict = torch.zeros([32000, 1, 256, 256])
remover_predict = torch.zeros([32000, 1, 256, 256])

for i in range(1000):
    with torch.no_grad():
        target_predict[i * 32:(i+1)*32] = goal(target[i * 32:(i+1)*32].repeat(1, 3, 1, 1).to("cuda")).clone()
        remover_predict[i * 32:(i+1)*32] = remover(input[i * 32:(i+1)*32].repeat(1, 3, 1, 1).to("cuda")).clone()
    print(i)

torch.save(target_predict, "dataset/all_1217/target.pt")
torch.save(remover_predict, "dataset/all_1217/current.pt")

goal = []
remover = []
target = []
input = []

action = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/action.pth")
action = action.view(32000, 6)

def rand_ints_nodup(a, b, k):
    ns = []
    while len(ns) < k:
        n = np.random.randint(a, b)
        if not n in ns:
            ns.append(n)
    nss = np.sort(np.array(ns))
    return nss

# num = np.array(range(24000))
nums = np.array(range(32000))

val_idx = rand_ints_nodup(0, 32000, 4000)
train_idx = np.delete(nums, val_idx)
# input = input.view(32000, 1, 224, 224)
# target = target.view(32000, 1, 224, 224)

torch.save(target_predict[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/all_1217/target_train.pt")
torch.save(target_predict[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/all_1217/target_val.pt")

torch.save(remover_predict[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/all_1217/current_train.pt")
torch.save(remover_predict[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/all_1217/current_val.pt")

torch.save(action[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/all_1217/action_train.pt")
torch.save(action[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/all_1217/action_val.pt")
