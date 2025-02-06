import torch
from PIL import Image
import numpy as np

current = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_ob.pt")
print(current.shape)
# for i in range(current.shape[0]):
#     for j in range(current.shape[1]):
#         for k in range(current.shape[2]):
            
#             image = current[i, j, k]* 255
#             image = image.to('cpu').detach().numpy().copy()
#             image = Image.fromarray((image).astype(np.uint8))
#             image.save(f"image/check/current_ob/{i}_{j}_{k}.png")

target = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/target.pt")
print(target.shape)
for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            image = target[i, j, k]* 255
            image = image.to('cpu').detach().numpy().copy()
            image = Image.fromarray((image).astype(np.uint8))
            image.save(f"image/check/target/{i}_{j}_{k}.png")