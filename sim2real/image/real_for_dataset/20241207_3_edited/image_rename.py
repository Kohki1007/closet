import os
import re

# 保存した画像ファイルがあるディレクトリを指定
directory = 'img'  # 画像が保存されているフォルダのパスに置き換えてください

# ファイル名の中の数値を取得するための正規表現
pattern = re.compile(r'rgb_image_(\d+)_\d+\.png')
# pattern = re.compile(r'depth_image_(\d+)_\d+\.png')

# 指定ディレクトリのファイルリストを取得し、パターンに一致するファイルのみを取得
files = [f for f in os.listdir(directory) if pattern.match(f)]
# ファイルを番号順にソート
# files.sort(key=lambda x: int(pattern.search(x).group(1)))

files.sort(key=lambda x: pattern.search(x).group(1))

# 各ファイルの名前を0から始まる連番に置き換える
for i, filename in enumerate(files):
    # 新しいファイル名を作成
    new_name = f'rgb_image_{i}.png'
    # new_name = f'depth_image_{i}.png'

    # 元のファイルパスと新しいファイルパス
    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_name)
    # ファイルをリネーム
    os.rename(old_path, new_path)
    print(f'Renamed {filename} to {new_name}')

print("すべてのファイル名が更新されました。")
