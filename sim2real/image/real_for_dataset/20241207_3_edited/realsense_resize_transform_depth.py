import os
from PIL import Image
from torchvision import transforms
import numpy as np

n = 120

# 画像を256x256にリサイズするtransform
resize_transform = transforms.Resize((256, 256))

# 読み込むフォルダと保存先フォルダのパス
input_folder = "img"  # 読み込む画像が保存されているフォルダのパス
output_folder = "resize"  # 保存先フォルダのパス

# 出力フォルダが存在しない場合、作成する
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(n):
    # depth画像の読み込み
    depth_filename = os.path.join(input_folder, f"depth_image_{i}.png")
    depth_image = Image.open(depth_filename)

    depth_size = depth_image.size
    print(f"Depth Image ({depth_filename}): {depth_image}")

    w, h = depth_image.size
    print(f"raw{i}: h={h}, w={w}")

    # 画像を256x256にリサイズ
    depth_resized = resize_transform(depth_image)
    w, h = depth_resized.size
    print(f"resized{i}: h={h}, w={w}")

    # NumPy配列に変換し、正規化
    depth_array = np.array(depth_resized).astype(np.float32)
    print(f"Depth array shape: {depth_array.shape}")
    max_val = depth_array.max()
    normalized_depth = depth_array / max_val if max_val > 0 else depth_array
    max_val = normalized_depth.max()

    # チャンネル追加（[1, H, W]に変換）
    normalized_depth = normalized_depth[np.newaxis, ...]

    normalized_depth_image = Image.fromarray((normalized_depth[0] * 255).astype(np.uint8))

    # 保存
    depth_filename_resize = os.path.join(output_folder, f"depth_image_resize_transform_{i}.png")
    normalized_depth_image.save(depth_filename_resize)
    print(f"Saved: {depth_filename_resize}")
