import numpy as np
import matplotlib.pyplot as plt
import glob
import torch

# x = np.array([87, 89, 89, 89, 76, 71, 39, 71, 89, 45, 40, 89, 30, 74.5, 47.5, 89, 44.5, 82, 89.5, 56.5, 89.5, 89, 50, 89, 89, 87, 90, 44.5, 62.5, 80.5, 25, 90, 55, 90, 90, 89.5, 71.5, 90, 90, 90, 48.5, 84, 90, 
#                  39, 90, 85.5, 50.5, 34.5, 75, 90, 89.5, 89.5, 88.5, 78, 85.5, 27.5, 29.5, 77.5, 69, 76, 71, 49, 90, 89.5, 73, 79, 89.5, 89.5, 90, 39.5, 49.5, 32.5, 79, 48, 34, 88, 62, 84.5, 39.5, 76.5, 
#                  90, 55, 80, 90, 32, 69.5, 42.5, 48, 89.5, 37, 89.5, 73, 45, 35, 75.5, 89.5, 90, 82, 65, 48])

search_string = "17"
file_pattern = "*.pt" 
result_list = []
for filename in glob.glob(file_pattern):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        if search_string in content:
            result_list.append(torch.load(f"{filename}"))

plt.hist(x, bins=9, edgecolor='black', range=(0, 90))
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

plt.savefig('/home/engawa/py_ws/visual_servo/src/network/images_sice/result.png')