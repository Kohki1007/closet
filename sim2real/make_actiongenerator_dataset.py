import torch
import numpy as np

action = torch.zeros(160, 5, 40, 6)
closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_rot.pt")
handle_trajectory_0 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/handle_trajectory_0_1217.pth")
handle_trajectory_1 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/handle_trajectory_1_1217.pth")

handle_trajectory = torch.cat([handle_trajectory_0.clone(), handle_trajectory_1.clone()], dim = 0)
torch.save(handle_trajectory, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/handle_trajectory.pth")
# print(closet_rot.shape)

for env in range(closet_rot.shape[0]):
    for position in range(closet_rot.shape[1]):
        for step in range(closet_rot.shape[2] - 1):
            start_point = closet_rot[env, position, step]

            action[env, position, step, :6] = handle_trajectory[env, position, int(start_point) + 4] - handle_trajectory[env, position, int(start_point)]
            # output[step, 6:] = handle_trajectory[int(start_point)]
            action[env, position, step, 2] = 0

            action[env, position, step, :3] /= torch.sqrt(torch.sum(action[env, position, step, :3]**2))
            action[env, position, step, 3:] /= torch.sqrt(torch.sum(action[env, position, step, 3:]**2))

torch.save(action, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/action.pth")



target1 = torch.load("dataset/dataset_generator_many/target_0_1214.pt")
target2 = torch.load("dataset/dataset_generator_many/target_1_1214.pt")

target = torch.cat([target1.clone(), target2.clone()], dim = 0)
torch.save(target, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/target.pt")

target1 = []
target2 = []

current_handle = torch.load("dataset/remover_sim/current_handle.pt")
current_st = torch.load("dataset/remover_sim/current_st.pt")

for i in range(2):
    currant_ob_element = torch.load(f"dataset/remover_sim/current_ob_{i}.pt")
    if i == 0:
        current_ob = currant_ob_element.clone()
    else:
        current_ob = torch.cat([current_ob, currant_ob_element], dim = 0)

curent = torch.cat([current_handle.clone(), current_st.clone(), current_ob.clone()], dim = 0)

torch.save(curent, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/current.pt")

# action = torch.load()

current_handle = []
current_st = []
current_ob = []

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
curent = curent.view(32000, 1, 224, 224)
target = target.view(32000, 1, 224, 224)
action = action.view(32000, 6)

torch.save(target[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/target_train.pt")
torch.save(target[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/target_val.pt")

torch.save(curent[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/current_train.pt")
torch.save(curent[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/current_val.pt")

torch.save(action[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/action_train.pt")
torch.save(action[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/action_val.pt")
