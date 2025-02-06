import torch
import sys 
import argparse
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms

from train_function_remover import train_function


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='goal_image_generator', help="学習するモデル名")
parser.add_argument("--lrt", type=float, default=0.0001, help="学習率")
parser.add_argument("--epoch", type=str, default=300000000000, help="エポック数")
parser.add_argument("--batch", type=str, default=32, help="バッチサイズ")
parser.add_argument("--weight_path", type=str, default="log/remover_sim_ur_2/weight/weights_last.pth", help="重みのパス")
parser.add_argument("--save_path", type=str, default="remover_sim_ur_3", help="保存先のパス")
parser.add_argument("--current_path", type=str, default="dataset_remover_sim/current_train.pt", help="入力１") ####### image1
parser.add_argument("--target_path", type=str, default="dataset_remover_sim/target_train.pt", help="入力１")
parser.add_argument("--action_path", type=str, default=None, help="アクション")

parser.add_argument("--eval", type=str, default=True, help="入力１")
parser.add_argument("--current_path_eval", type=str, default="dataset_remover_sim/current_val.pt", help="入力１") ####### image1
parser.add_argument("--target_path_eval", type=str, default="dataset_remover_sim/target_val.pt", help="入力１")
parser.add_argument("--action_path_eval", type=str, default=None, help="入力１")

args = parser.parse_args()


train_base = train_function(args.model, args.weight_path, args.epoch, args.lrt, args.batch)
train_base.make_dir(args.save_path)

train_dataset = train_base.make_dataset(args.current_path, args.target_path, args.action_path, args.batch)

if args.eval:
    # print(args.eval)
    eval_dataset = train_base.make_dataset_eval(args.current_path_eval, args.target_path_eval, args.action_path_eval)
    print(args.eval)

best_loss = 100
train_history = []
valid_history = []

noise_level = 0

for epoch in range(args.epoch):
    
    print(f'epoch === {epoch}')

    # train_base.save_sample_image(args.model, args.save_path, epoch, args.eval)

    # forループ内で使う変数と、エポックごとの値リセット
    train_loss = 0.  
    train_accuracy_5 = 0.  
    train_accuracy_1 = 0.

    val_loss = 0.  
    val_all_loss = 0.
    val_accuracy_5 = 0.  
    val_accuracy_1 = 0.

    accuracy_1_all = 0
    accuracy_1_handle = 0
    accuracy_1_withouthandle = 0

    with tqdm(enumerate(train_dataset),
              total=len(train_dataset)) as pbar_loss:
        for i, (current, target) in pbar_loss:
            loss, accuracy = train_base.train_step_remover(current, target, epoch)

            # 取得した損失値と正解率を累計値側に足していく  
            train_loss += loss
            train_accuracy_1 += accuracy[0] 
            train_accuracy_5 += accuracy[1]  


    # train_base.plot_grad(epoch)
    n = epoch + 1                             # 処理済みのエポック数
    train_loss /= (i+1) 
    train_accuracy_1 /= (i+1) 
    train_accuracy_5 /= (i+1) 
    train_history.append(train_loss.to('cpu').detach().numpy().copy())

    
    


    if args.eval:
        # train_base.save_sample_image(args.save_path, epoch, args.eval)

        with tqdm(enumerate(train_dataset),
              total=len(train_dataset)) as pbar_loss:
            for j, (current, target) in pbar_loss:
                val_loss, val_accuracy = train_base.eval_step_remover(current, target, epoch)

                val_all_loss += val_loss  
                val_accuracy_1 += val_accuracy[0]  
                val_accuracy_5 += val_accuracy[1]  

        val_all_loss /= (j+1) 
        val_accuracy_1 /= (j+1) 
        val_accuracy_5 /= (j+1) 
    
        valid_history.append(val_all_loss.to('cpu').detach().numpy().copy())

        train_base.save_sample_image(args.model, args.save_path, epoch, args.eval)

    if (epoch+1)%5 == 0:
        epochs = len(train_history)
        plt.clf()
        mean = np.mean(train_history)
        min = np.min(train_history)
        plt.plot(range(epochs), train_history, marker='.', label='loss (Training data)')
        if args.eval:
            plt.plot(range(epochs), valid_history, marker='.', label='loss (Validation data)')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + args.save_path + f'/loss/{epoch}.png')
        plt.clf()

    if (epoch+1)%20 == 0 :
        train_base.modify_oprimazer(epoch)
        
    if (epoch+1)%10 == 0:
        train_base.save_weight(args.save_path, epoch)

    if train_loss <= best_loss:
        train_base.save_weight(args.save_path, epoch, last = None, best = True)

    if val_accuracy_1 > 0.95:
        noise_level += 0.01
        train_base.save_weight(args.save_path, epoch, last = None, best = True, noise = noise_level)
        train_base.reset_optimizer()

    train_base.save_weight(args.save_path, epoch, last = True)

    f = open("/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + args.save_path  + f'/data.txt', 'a')

    f.write(f'epoch {epoch} loss {train_loss} train_accuracy_1: {train_accuracy_1:.5f}, train_accuracy_5: {train_accuracy_5:.5f}\n')

    print(f'[Epoch {epoch+1:3d}/{args.epoch}]' \
    f' noise_level : {noise_level}, train_loss: {train_loss:.5f}'\
    f' train_accuracy_1: {train_accuracy_1:.5f}, train_accuracy_5:: {train_accuracy_5:.5f}')

    if args.eval:
        print(f' val_loss: {val_all_loss:.5f}'\
        f' val_accuracy_1: {val_accuracy_1:.5f}, val_accuracy_5:: {val_accuracy_5:.5f}')

print('Finished Training')


