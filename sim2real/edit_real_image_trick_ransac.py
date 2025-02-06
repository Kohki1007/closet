import torch
from PIL import Image
import os
import numpy as np
import pyransac3d as pyrsc

input_folder = "image/eval/trick/input/"
depth_filename = os.path.join(input_folder, f"0.png")
image = torch.from_numpy(np.array(Image.open(depth_filename)))
image = image.type(torch.float64)
image /= 255
image = torch.where(image > 0.8, 1., image)

##### ドア部分を抽出
# image = torch.where(image < 0.8, 0., image)

indices_top = torch.where(image[0, :] < 0.8)

top_left = torch.min(indices_top[0])
top_right = torch.max(indices_top[0])

indices_bottom = torch.where(image[223, :] < 0.8)
print(indices_bottom)
bottom_left = torch.min(indices_bottom[0])
bottom_right = torch.max(indices_bottom[0])

left = int((top_left + bottom_left)/2)
left = bottom_left
right = int((top_right + bottom_right)/2)
image[:, :(left)] = 1.0
image[:, (right+1):] = 1.0
width = right - left
print(width)

##### ドア部分を点群に変換
point_cloud = torch.zeros(224 * (right - left), 3)
for y in range(right - left):
    y_number = y + left
    for x in range(224):
        ind = x + y * 224
        point_cloud[ind] = torch.tensor([x, y_number, image[x, y_number]])


##### ransac を実行
point_cloud = point_cloud.to('cpu').detach().numpy().copy()
plano1 = pyrsc.Plane()
best_eq, best_inliers = plano1.fit(point_cloud, 0.01)
inliers = point_cloud[best_inliers]
for i in range(len(best_inliers)):
    x = int(inliers[i, 0])
    y = int(inliers[i, 1])
    image[x, y] = 0.2


##### 任意の開閉角度になるように画像を加工 

save_image = image.to('cpu').detach().numpy().copy() * 255
save_image = Image.fromarray((save_image).astype(np.uint8))
# depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
save_image.save(f"image/eval/trick/test/inliers.png")