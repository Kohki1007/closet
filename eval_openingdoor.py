import torch
import torch.nn as nn
import numpy as np
# from torchviz import make_do
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from encoder import Autoencoder, Autoencoder_color, Autoencoder_attension, Autoencoder_rgbd_attension
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from livelossplot import PlotLosses
import torch.nn.init as init
from PIL import Image

import sys 

base = 'src/network/predict_opening_door/logs/rgbd_attension'
args = sys.argv

args = [0, "continue", "rgbd_attension"]

BATCH_SIZE = 32 #2の11乗
LEARNING_RATE = 0.0001  # 学習率： 0.03
REGULARIZATION = 0.03  # 正則化率： 0.03


best_accuracy = 0
device2 = torch.device("cuda")
device2 = torch.device("cpu")

path_train_input = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/train/input1.pth"
path_train_output = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/train/output1.pth"
path_val_input = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/test/input1.pth"
path_val_output = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/test/output1.pth"
path_train_input2 = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/train/input2.pth"
path_train_output2 = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/train/output2.pth"
path_val_input2 = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/test/input2.pth"
path_val_output2 = f"src/network/predict_opening_door/dataset/predict_opening_wall/{args[2]}/test/output2.pth"

# train_x = torch.cat((torch.load(path_train_input).to(device2), torch.load(path_train_input2).to(device2)), dim=1)
# train_y = torch.cat((torch.load(path_train_output).to(device2), torch.load(path_train_output2).to(device2)), dim=1)

# val_x = torch.cat((torch.load(path_val_input).to(device2), torch.load(path_val_input2).to(device2)), dim=1)
# val_y = torch.cat((torch.load(path_val_output).to(device2), torch.load(path_val_output2).to(device2)), dim=1)

train_x = torch.load(path_train_input).to(device2)
train_y = torch.load(path_train_output).to(device2)

val_x = torch.load(path_val_input).to(device2)
val_y = torch.load(path_val_output).to(device2)

# train_x = train_x.to(torch.float32)
# train_y = (train_y).to(torch.float32)
val_x = val_x.to(torch.float32)
val_y = (val_y).to(torch.float32)


dataset_train = TensorDataset(train_x, train_y)  # 訓練用
dataset_valid = TensorDataset(val_x, val_y)  # 精度検証用
# 
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=int(BATCH_SIZE/4), shuffle=True)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)


# NMS_start_merion = CustomLoss()
loss_mae = nn.L1Loss()
loss_mse = nn.MSELoss()
# par = torch.load('runs/160_層減少_1_020100/weight/weights29975.pth', map_location='cuda')
if args[2] == 'depth':
    model = Autoencoder()
elif args[2] == 'attension':
    model = Autoencoder_attension()
elif args[2] == 'rgbd_attension':
    model = Autoencoder_rgbd_attension()
else:
    model = Autoencoder_color()

# print(model)
# print(model.device)
def train_step(train_X, train_y):
    train_X = train_X.to('cuda')
    train_y = train_y.to('cuda')
    train_X = train_X.view(-1, 5, 480, 640)
    train_y = train_y.view(-1, 1, 480, 640)
    # train_y = train_y.view(-1, 5, 480, 640)
    train_y = train_y[:, 3]
    train_y = train_y.unsqueeze(dim=1)
    # 訓練モードに設定
    model.train()

    # train_X = torch.tensor([[ 0.53438,  0.62222,  0.40000,  0.62222, -0.50611, -0.02975]], device='cuda:0')

    # フォワードプロパゲーションで出力結果を取得
    #train_X                # 入力データ
    pred_y = model(train_X) # 出力結果

    #train_y                # 正解ラベル

    # 出力結果と正解ラベルから損失を計算し、勾配を求める
    optimizer.zero_grad()   # 勾配を0で初期化（※累積してしまうため要注意）
    loss = loss_mae(pred_y, train_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得
    loss.backward()   # 逆伝播の処理として勾配を計算（自動微分）

    # 勾配を使ってパラメーター（重みとバイアス）を更新
    optimizer.step()   # 勾配を0で初期化（※累積してしまうため要注意）
    
    # 正解数を算出
    with torch.no_grad(): # 勾配は計算しないモードにする
        mae = loss_mae(pred_y, train_y)
        mse = loss_mse(pred_y, train_y) # 正解数を合計する
        msel = torch.sqrt(mse)

    df = torch.abs(pred_y - train_y)
    train_per_xyz_1 = torch.sum(torch.where(df<0.01, 1, 0))/torch.numel(df)
    train_per_xyz_5 = torch.sum(torch.where(df<0.05, 1, 0))/torch.numel(df)

    train_X = train_X.to('cpu')
    train_y = train_X.to('cpu')
    return  (loss, mae.item(), mse.item(), msel.item(),  train_per_xyz_1, train_per_xyz_5)    # ※item()=Pythonの数値

def valid_step(valid_X, valid_y):
    # valid_X = valid_X.to('cuda')
    # valid_y = valid_y.to('cuda')
    valid_X = train_x[0].to('cuda')
    valid_y = train_y[0].to('cuda')
    valid_X = valid_X.view(-1, 5, 480, 640)
    # # valid_X = valid_X[:, [0, 2, 1, 3]]
    valid_y = valid_y.view(-1, 1, 480, 640)
    output_image = valid_y
    # valid_y = valid_y[:, 3]
    # valid_y = valid_y.unsqueeze(dim=1)
    # # valid_y = valid_y[:, [2, 1, 0, 3]]
    # color_image = valid_X[0, :3] * 255
    depth_input = valid_X[0, 3] * 200
    depth_image = output_image[0, 0] * 200
    # color_output = output_image[0, :3] * 255
    # color_image = color_image.permute(1, 2, 0)
    # color_image = color_image.to('cpu').detach().numpy().copy()
    # color = Image.fromarray((color_image).astype(np.uint8))
    # color_output = color_output.permute(1, 2, 0)
    # color_output = color_output.to('cpu').detach().numpy().copy()
    # color_output = Image.fromarray((color_output).astype(np.uint8))
    depth_image = depth_image.to('cpu').detach().numpy().copy()
    depth_correct = Image.fromarray((depth_image).astype(np.uint8))
    depth_input = depth_input.to('cpu').detach().numpy().copy()
    depth_input = Image.fromarray((depth_input).astype(np.uint8))
    # color.save("src/network/predict_opening_door/dataset/predict_opening/images/color.png")
    # color_output.save("src/network/predict_opening_door/dataset/predict_opening/images/color_output.png")
    depth_correct.save("src/network/predict_opening_door/dataset/predict_opening/images/depth_correct_0722.png")
    depth_input.save("src/network/predict_opening_door/dataset/predict_opening/images/depth_input_0722.png")
    # 評価モードに設定（※dropoutなどの挙動が評価用になる）
    model.eval()

    # フォワードプロパゲーションで出力結果を取得
    #valid_X                # 入力データ
    pred_y = model(valid_X) # 出力結果

    depth_predict = pred_y[0, 0] * 200
    depth_predict = depth_predict.to('cpu').detach().numpy().copy()
    depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
    depth_predict.save("src/network/predict_opening_door/dataset/predict_opening/images/depth_predict_0722.png")


    with torch.no_grad(): 
        loss = loss_mae(pred_y, valid_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得
    # ※評価時は勾配を計算しない
    # 正解数を算出
    with torch.no_grad(): # 勾配は計算しないモードにする
        mae = loss_mae(pred_y, valid_y)
        mse = loss_mse(pred_y, valid_y) # 正解数を合計する
        msel = torch.sqrt(mse)

    # 損失と正解数をタプルで返す
    df = torch.abs(pred_y - valid_y)
    val_per_xyz_1 = torch.sum(torch.where(df<0.01, 1, 0))/torch.numel(df)
    val_per_xyz_5 = torch.sum(torch.where(df<0.05, 1, 0))/torch.numel(df)
    valid_X = valid_X.to('cpu')
    valid_y = valid_y.to('cpu')
    return  (loss, mae.item(), mse.item(), msel.item(),  val_per_xyz_1, val_per_xyz_5)  # ※item()=Pythonの数値

if args[1] == "continue":
    print("oo")
    par = torch.load(base + f'/weight/weights_last.pth', map_location='cuda')
    model.load_state_dict(par)
    model.to('cuda')

else:
    model.apply(weights_init)
    model.to('cuda')
    

# 定数（学習／評価時に必要となるもの）
EPOCHS = 30000000           # エポック数： 100
epochs = range(EPOCHS)
# 変数（学習／評価時に必要となるもの）
avg_loss = 0.0           # 「訓練」用の平均「損失値」
avg_mape = 0.0            # 「訓練」用の平均「正解率」
avg_val_loss = 0.0       # 「評価」用の平均「損失値」
avg_val_mape = 0.0        # 「評価」用の平均「正解率」

# オプティマイザを作成（パラメーターと学習率も指定）
optimizer = optim.AdamW(           
    model.parameters(),          # 最適化で更新対象のパラメーター（重みやバイアス）
    lr=LEARNING_RATE,            # 更新時の学習率
    weight_decay=REGULARIZATION) # L2正則化（※不要な場合は0か省略）

# 損失の履歴を保存するための変数
train_history = []
valid_history = []
train_mape = []
valid_mape = []
liveloss = PlotLosses()

y_train = np.zeros(EPOCHS)
y_val = np.zeros(EPOCHS)

logs = {}
best_epoch = 0
best_loss = 10

for epoch in epochs:
    
    print(f'epoch === {epoch}')

    # forループ内で使う変数と、エポックごとの値リセット
    total_loss = 0.0     # 「訓練」時における累計「損失値」
    total_mae = 0
    total_mse = 0.0      # 「訓練」時における累計「正解数」
    total_msel = 0.0 
    total_val_loss = 0.0 # 「評価」時における累計「損失値」
    total_val_mae = 0
    total_val_mse = 0.0  # 「評価」時における累計「正解数」
    total_val_msel = 0.0
    total_train = 0      # 「訓練」時における累計「データ数」
    total_valid = 0        # 「評価」時における累計「データ数」
    train_xyz_1 = 0
    train_xyz_5 = 0
    val_xyz_1 = 0
    val_xyz_5 = 0

    # with tqdm(enumerate(loader_train),
    #           total=len(loader_train)) as pbar_loss:
    #     for i, (train_X, train_y) in pbar_loss:
    #         # print(train_X)
    #         # print(train_y.type)
    #         # 【重要】1ミニバッチ分の「訓練」を実行
    #         # print(train_X.shape)
    #         loss, mae, mse, msel,  train_per_xyz_1, train_per_xyz_5 = train_step(train_X, train_y)

    #         # 取得した損失値と正解率を累計値側に足していく
    #         total_loss += loss          # 訓練用の累計損失値
    #         total_mae += mae
    #         total_mse += mse            # 訓練用の累計正解数
    #         total_msel += msel
    #         total_train += len(train_y) # 訓練データの累計数
    #         train_xyz_1 += train_per_xyz_1
    #         train_xyz_5 += train_per_xyz_5
    #         # train_eular += train_per_eular

    with tqdm(enumerate(loader_valid),
              total=len(loader_valid)) as pbar_loss:
        for j, (valid_X, valid_y) in pbar_loss:
            # 【重要】1ミニバッチ分の「評価（精度検証）」を実行
            val_loss, val_mae, val_mse, val_msel, val_per_xyz_1 , val_per_xyz_5= valid_step(valid_X, valid_y)

            # 取得した損失値と正解率を累計値側に足していく
            total_val_loss += val_loss  # 評価用の累計損失値
            total_val_mae += val_mae
            total_val_mse += val_mse    # 評価用の累計正解数
            total_val_msel += val_msel 
            total_valid += len(valid_y)# 訓練データの累計数
            val_xyz_1 += val_per_xyz_1
            val_xyz_5 += val_per_xyz_5
            # val_eular += val_per_eular

    # ミニバッチ単位で累計してきた損失値や正解率の平均を取る
    n = epoch + 1                             # 処理済みのエポック数
    avg_loss = total_loss / (i + 1)                # 訓練用の平均損失値
    avg_mse = total_mse / (i + 1)           # 訓練用の平均正解率
    avg_mae = total_mae /(i + 1)    
    avg_msel = total_msel / (i + 1)    
    train_xyz_1 = train_xyz_1/(i + 1)  
    train_xyz_5 = train_xyz_5/(i + 1)
    # train_eular = train_eular/(i + 1)  
    avg_val_loss = total_val_loss/(j+1)        # 訓練用の平均損失値
    avg_val_mse = total_val_mse/(j+1)   # 訓練用の平均正解率
    avg_val_mae = total_val_mae/(j+1)   
    avg_val_msel = total_val_msel/(j+1) 
    val_xyz_1 = val_xyz_1/(j+1) 
    val_xyz_5 = val_xyz_5/(j+1) 
    # val_eular = val_eular/(j+1) 
    
    
    # print(avg_loss)
    # print(avg_val_loss)
    # グラフ描画のために損失の履歴を保存する
    train_history.append(avg_loss.to('cpu').detach().numpy().copy())
    valid_history.append(avg_val_loss.to('cpu').detach().numpy().copy())
    train_mape.append(avg_mse)
    valid_mape.append(avg_val_mse)
    y_train[epoch] = avg_loss
    y_val[epoch] = avg_val_loss
    # 損失や正解率などの情報を表示
    print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \
        f' loss: {avg_loss:.5f}, mae: {avg_mae:.5f}' \
        f' mse: {avg_mse:.5f}, msel: {avg_msel:.5f}' \
        f' val_loss: {avg_val_loss:.5f}, val_mape: {avg_val_mae:.5f}'\
        f' val_mse: {avg_val_mse:.5f}, val_msel: {avg_val_msel:.5f}'\
         f' train_xyz_1: {train_xyz_1:.5f}, train_xyz_5: {train_xyz_5:.5f}'\
        f' val_xyz_1: {val_xyz_1:.5f}, val_xyz_5: {val_xyz_5:.5f}' \
        f' best accuracy: {best_accuracy:.5f}'  \
        f' best epoch: {best_epoch:.5f}'    )
    # print(f'train_history = {train_history}, val_history = {valid_history}')
    # print(train_history)
    # print(train_history)

    if (epoch+1)%5 == 0:
        epochs = len(train_history)
        plt.clf()
        mean = np.mean(train_history)
        min = np.min(train_history)
        plt.plot(range(epochs), train_history, marker='.', label='loss (Training data)')
        plt.plot(range(epochs), valid_history, marker='.', label='loss (Validation data)')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # if mean > 0.5:
        #     plt.ylim(0, mean)
        # else:
        #     plt.ylim(0, 0.015)
        plt.savefig(base  +f'/loss/{epoch}.png')
        plt.clf()

    if (epoch+1)%30 == 0 and LEARNING_RATE > 0.000001:
        LEARNING_RATE /= 2
        optimizer = optim.AdamW(           
            model.parameters(),          # 最適化で更新対象のパラメーター（重みやバイアス）
            lr=LEARNING_RATE,            # 更新時の学習率
            weight_decay=REGULARIZATION)

    # logs['loss'] = train_history[epoch]
    # logs['val_loss'] = valid_history[epoch]

    # logs['mape'] = train_mape[epoch]
    # logs['val_mape'] = valid_mape[epoch]


    # liveloss.update(logs)
    # liveloss.send()
    
    # x = MPC.get_param()
    if (epoch+1)%25 == 0:
        save_path = base + f'/weight/weights{epoch}.pth'
        torch.save(model.state_dict(), save_path) 

    if train_xyz_1 > best_accuracy:
        best_accuracy = train_xyz_1
        best_epoch = epoch
        save_path = base + f'/weight/weights_best.pth'
        torch.save(model.state_dict(), save_path) 

    if avg_loss <= best_loss:
        best_loss = avg_loss
        # best_accuracy = train_xyz
        # best_epoch = epoch
        # save_path = base + f'/weight/weights_best.pth'
        # torch.save(model.state_dict(), save_path)

        save_path = base + f'/weight/weights_last.pth'
        torch.save(model.state_dict(), save_path) 

    f = open(base + f'data.txt', 'a')

    f.write(f'epoch {epoch} loss {avg_loss} accuracy {train_xyz_1} {train_xyz_5} val_accuracy {val_xyz_1} {val_xyz_5}\n')

print('Finished Training')
# print(model.state_dict())
# print(model.state_dict())  # 学習後のパラメーターの情報を表示