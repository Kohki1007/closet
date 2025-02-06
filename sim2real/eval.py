import torch
import time
from model import ivis
from torchvision import transforms

# input_dataset_1 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/current_train_RGB_2.pt")
input_data = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/current_test_RGB_2.pt")
# input = torch.cat((input_dataset_1, input_dataset_2), dim = 0)

# output_dataset_1 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/target_train_2.pt")
# output_dataset_2 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/target_test_2.pt")
# output = torch.cat((output_dataset_1, output_dataset_2), dim = 1)

# rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_rot.pt")

# print(rot.shape)

# repeat = input // 50
# # for i in range(repeat):

model = ivis()
transform = transforms.Resize((224, 224))
input_data = transform(input_data[0].unsqueeze(0))
# print(input_data.shape)
# predict = model()
model = model.to("cuda")

with torch.no_grad():
    start = time.time()
    predict = model.goal_image_generator(input_data.to("cuda"))
    end = time.time() 

    print(start - end)

