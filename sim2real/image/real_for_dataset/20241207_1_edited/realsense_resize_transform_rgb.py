import os
from PIL import Image
from torchvision import transforms

n = 79

# 画像を256x256にリサイズするtransform
resize_transform = transforms.Resize((256, 256))

# 読み込むフォルダと保存先フォルダのパス
input_folder = "img"  # 読み込む画像が保存されているフォルダのパス
output_folder = "resize"  # 保存先フォルダのパス

# 出力フォルダが存在しない場合、作成する
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(n):
    # RGB画像の読み込み
    rgb_filename = os.path.join(input_folder, f"rgb_image_{i}.png")
    rgb_image = Image.open(rgb_filename)

    w, h = rgb_image.size
    print(f"raw{i}: h={h}, w={w}")

    # 画像を256x256にリサイズ
    rgb_resized = resize_transform(rgb_image)
    w, h = rgb_resized.size
    print(f"resized{i}: h={h}, w={w}")

    # 保存
    rgb_filename_resize = os.path.join(output_folder, f"rgb_image_resize_transform_{i}.png")
    rgb_resized.save(rgb_filename_resize)
