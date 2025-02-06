import torch
import numpy as np

# closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/real_近い/closet_rot.pt")
# handle_trajectory = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/trajectory_0.pth")
# handle_trajectory_1 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/trajectory_1.pth")
# handle_trajectory = torch.cat([handle_trajectory, handle_trajectory_1], dim = 0)
# action = torch.zeros(160, 5, 40, 6)

# # print(closet_rot.shape)

# for env in range(closet_rot.shape[0]):
#     for position in range(closet_rot.shape[1]):
#         for step in range(closet_rot.shape[2]-1):
#             start_point = closet_rot[env, position, step]

#             if int(start_point) + 4 < 181:
#                 action[env, position, step, :6] = handle_trajectory[env, position, int(start_point) + 4] - handle_trajectory[env, position, int(start_point)]
#             elif int(start_point) + 4 > 180:
#                 print(step)
#                 action[env, position, step, :6] = handle_trajectory[env, position, 180] - handle_trajectory[env, position, int(start_point)]
#             # output[step, 6:] = handle_trajectory[int(start_point)]
#             action[env, position, step, 2] = 0

#             action[env, position, step, :3] /= torch.sqrt(torch.sum(action[env, position, step, :3]**2))
#             action[env, position, step, 3:] /= torch.sqrt(torch.sum(action[env, position, step, 3:]**2))

# torch.save(action, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/action_withouthandle.pth")


# target1 = torch.load("dataset/real__近い/target_withouthandle_0.pt")
# target2 = torch.load("dataset/real__近い/target_withouthandle_1.pt")

# target = torch.cat([target1.clone(), target2.clone()], dim = 0)

# torch.save(target, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/targets_withouthandle.pt")

# target1 = []
# target2 = []

# current_handle = torch.load("dataset/real__近い/current_withouthandle_normal.pt")
# current_st = torch.load("dataset/real__近い/current_withouthandle_st.pt")

# for i in range(10):
#     number = 100 + i * 6
#     currant_ob_element = torch.load(f"dataset/real__近い/current_withouthandle_ob_{number}.pt")
#     if i == 0:
#         current_ob = currant_ob_element.clone()
#     else:
#         current_ob = torch.cat([current_ob, currant_ob_element], dim = 0)

# curent = torch.cat([current_handle.clone(), current_st.clone(), current_ob.clone()], dim = 0)

# torch.save(curent, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/current_withouthandle.pt")

current = []

target1 = torch.load("dataset/real__近い/target_0.pt")
target2 = torch.load("dataset/real__近い/target_1.pt")

target = torch.cat([target1.clone(), target2.clone()], dim = 0)
# print(target2.shape)

torch.save(target, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/targets.pt")

target1 = []
target2 = []

current_handle = torch.load("dataset/real__近い/current_normal.pt")
current_st = torch.load("dataset/real__近い/current_st.pt")

for i in range(10):
    number = 100 + i * 6
    currant_ob_element = torch.load(f"dataset/real__近い/current_ob_{number}.pt")
    if i == 0:
        current_ob = currant_ob_element.clone()
    else:
        current_ob = torch.cat([current_ob, currant_ob_element], dim = 0)

curent = torch.cat([current_handle.clone(), current_st.clone(), current_ob.clone()], dim = 0)

torch.save(curent, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/current.pt")

# current_handle = []
# current_st = []
# current_ob = []

# def rand_ints_nodup(a, b, k):
#     ns = []
#     while len(ns) < k:
#         n = np.random.randint(a, b)
#         if not n in ns:
#             ns.append(n)
#     nss = np.sort(np.array(ns))
#     return nss

# # num = np.array(range(24000))
# nums = np.array(range(32000))

# val_idx = rand_ints_nodup(0, 32000, 4000)
# train_idx = np.delete(nums, val_idx)
# curent = curent.view(32000, 1, 224, 224)
# # target = target.view(32000, 1, 224, 224)

# # torch.save(target[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/target_train.pt")
# # torch.save(target[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/target_val.pt")

# torch.save(curent[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/current_train.pt")
# torch.save(curent[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/current_val.pt")
