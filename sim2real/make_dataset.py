import torch
import numpy as np

input = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_generator_many/current.pt")

output0 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_many/output_0_1214.pt")
output1 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_many/output_1_1214.pt")

output = torch.cat([output0.clone(), output1.clone()], dim = 0)

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

val_idx = val_idx.tolist()
train_idx = train_idx.tolist()
# print(current.shape)
current = input.view(-1, 1, 224, 224)
current_train = current[train_idx]
current_val = current[val_idx]

torch.save(current_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_many/input_train.pt")
torch.save(current_val, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_many/input_val.pt")

current_train = []
current_val = []
current = []

target = output.view(-1, 1, 224, 224)
# print(target.shape)
target_train = target[train_idx]
target_val = target[val_idx]

torch.save(target_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_many/output_train.pt")
torch.save(target_val, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_many/output_val.pt")