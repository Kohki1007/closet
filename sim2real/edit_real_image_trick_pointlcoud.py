import torch
from PIL import Image
import os
import numpy as np
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
from torchvision import transforms

# カメラ内部パラメータ
fx = 908.1617431640625  # 焦点距離 x
fy = 906.48297  # 焦点距離 y
cx = 637.798  # 画像中心 x
cy = 371.0213  # 画像中心 y
transform = transforms.Resize((224, 224))

depth_filename = "dataset/trick/trick_eval/img/depth_image_0.png"
image = torch.from_numpy(np.array(Image.open(depth_filename)))
noemalized_depth = image/torch.max(image)

indices_top = torch.where(noemalized_depth[0, :] < 0.8)

top_left = torch.min(indices_top[0])
top_right = torch.max(indices_top[0])

indices_bottom = torch.where(noemalized_depth[-1, :] < 0.8)
# print(indices_bottom)
bottom_left = torch.min(indices_bottom[0])
bottom_right = torch.max(indices_bottom[0])

left = int((top_left + bottom_left)/2)
left = bottom_left
right = int((top_right + bottom_right)/2)
right = bottom_right

image[:, :(left)] = 0.0
image[:, (right+1):] = 0.0

door_part = image[:, (left):(right+1)]
# image = transform(image.unsqueeze(0))
# image = image.squeeze()

# 画像の高さと幅
height, width = image.shape
# print(height, width)

# 画像上の各ピクセル座標を作成
v = torch.tensor(range(width))
u = torch.tensor(range(height))
u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
# print(u_grid)
# print(v_grid)

# 深度値を取得（必要に応じて単位変換する）
d = image.type(torch.float64)  # 例えば、メートル単位に変換済みの場合

# 各画素から3D座標を計算
X = (u_grid - cx) * d/ fx
Y = (v_grid - cy) * d / fy
Z = d

# print(X)
# print(X.shape)
# print(Y.shape)
# print(Z.shape)


###ドア部分を抽出
X_door = X[:, (left):(right+1)]
Y_door = Y[:, (left):(right+1)]
Z_door = Z[:, (left):(right+1)]
# print(X)

# 点群の作成
points = torch.stack((X_door, Y_door, Z_door), dim=-1)  # shape: (H, W, 3)
points = points.reshape(-1, 3)         # shape: (H*W, 3)

#RANSAC

point_cloud = points.to('cpu').detach().numpy().copy()
plano1 = pyrsc.Plane()
best_eq, best_inliers = plano1.fit(point_cloud, 0.01)
print(best_eq)
inliers = point_cloud[best_inliers]





###### ドア部分の平面の式を算出成功
## ドアの横幅を算出
left_point = (left - cx) * d[-1, left] / fx
right_point = (right - cx) * d[-1, right] / fx
door_wide = torch.abs(left_point - right_point)
fixdoor_wide = door_wide // 3

cols_with_target = ((points[:, 1].unsqueeze(0) < left_point) | (points[:, 1].unsqueeze(0) > right_point)).any(dim=0)

cols_to_keep = ~cols_with_target
points_withoutdoor = points[cols_to_keep]
###### 開閉角度に応じて３次元点群を変更
## 細かい点群でドアを構成
door_u = torch.linspace(left + fixdoor_wide, right, int(fixdoor_wide*2) * 4)
door_v = torch.linspace(0, 719, 2880)

new_u_grid, new_v_grid = torch.meshgrid(u, v, indexing='ij')
print(new_u_grid.shape, new_v_grid.shape)
d_door = (new_u_grid * best_eq[0] + new_v_grid * best_eq[1] + best_eq[3]) / best_eq[2]

X_newdoor = (new_u_grid - cx) * d_door/ fx
Y_newdoor = (new_v_grid - cy) * d_door/ fy
Z_newdoor = d_door



# ドア部分を回転させる
rotation_center = 

# 隙間にはmax地を挿入

## 原点から直線状に点が並ぶときは一番近いものを採用

## 深度画像用の粗さに修正


# z = points_withoutdoor[:, 0]
# y = points_withoutdoor[:, 1]
# x = points_withoutdoor[:, 2]

# # 図を作成し、3D Axes を追加
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # 点群の散布図を描画
# sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=10)

# # 軸ラベルとタイトル
# ax.set_title("3D 点群")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# # plt.savefig("image/eval/trick/test/pointcloud.png")
# plt.show()
