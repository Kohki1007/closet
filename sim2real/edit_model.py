import torch
import clip
from PIL import Image
import torch.nn as nn

from model import ivis

device = "cuda"

model, preprocess = clip.load("ViT-B/32", device=device)
print(model)

torch.save(model.visual.state_dict(), "src/network/sim2real/log/base_visual_weight.pt")

weight = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/log/base_visual_weight.pt", map_location='cuda')
positional_weight = weight["positional_embedding"].clone()
weight["positional_embedding"] = (torch.rand(65, 768) * 2 - 1)/10
weight["positional_embedding"][:50] = positional_weight
weight["conv1.weight"] = weight["conv1.weight"][:, :2]

torch.save(weight, "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/edit_visual_weight.pt")

# new_model = ivis()
# weight = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/log/base_weight.pt", map_location='cuda')
model.visual.load_state_dict(weight)
model.visual.conv1 = nn.Conv2d(2, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
scale = 0.03608439182435161
width = 768
model.positional_embedding = nn.Parameter(scale * torch.randn((65, width)))


print("aa")

# self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         # check = torch.load("/home/engawa/.cache/clip/ViT-B-32.pt")
#         # print(check)
#         self.encoder, preprocess = clip.load("ViT-B/32", device=self.device)
#         # conv = model.visual.conv1
#         self.encoder.visual.conv1 = nn.Conv2d(2, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
#         # self.encoder = model.encode_image()
#         scale = 0.03608439182435161
#         width = 768
#         self.encoder.positional_embedding = nn.Parameter(scale * torch.randn((65, width)))

#         torch.save(self.encoder.state_dict(), "src/network/sim2real/log/base_weight.pt")
#         print("aa")