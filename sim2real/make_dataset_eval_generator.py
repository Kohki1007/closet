import torch
import numpy as np

# for i in range(5):
#     currant_element = torch.load(f"dataset/eval/generator_accuracy/current_{i}.pt")
#     if i == 0:
#         current = currant_element.clone()
#     else:
#         current = torch.cat([current, currant_element], dim = 0)

for i in range(5):
    target_element = torch.load(f"dataset/eval/generator_accuracy/target_{i}.pt")
    if i == 0:
        target = target_element.clone()
    else:
        target = torch.cat([target, target_element], dim = 0)

current_element = []
target_element = []

target = target.view(-1, 180, 256, 256)
# current = current.view(-1, 180, 224, 224)

# torch.save(current, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/eval/generator_accuracy/current.pt")
torch.save(target, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/eval/generator_accuracy/target.pt")
