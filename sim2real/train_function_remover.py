import torch
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

from model import ivis, goal_image_generator, manipulator_remover, action_generator
from loss import goal_image_loss, action_loss
import cv2
import torchvision.transforms.functional as F
import torch.nn.functional as ff

class train_function():
    def __init__(self, train_subject = "goal_image_generator", 
                load_path = None, epoch = 300000, lerning_rate = 0.005, batch_size = 16):
        self.train_subject = train_subject
        self.device = "cuda"

        ################## model の指定         
        self.model = manipulator_remover()
        # self.model = ivis()
        self.model = self.model.to(self.device)
        # weight = torch.load("log/edit_visual_weight.pt", map_location='cuda')
        # weight = torch.load("log/edit_visual_weight.pt", map_location='cuda')
        # print(self.model)
        # print(weight)
        # self.model.goal_image_generator.encoder.load_state_dict(weight)
        self.model = self.model.to("cuda")

        # for name, param in self.model.goal_image_generator.named_parameters():
        #     print(f"Parameter: {name}, requires_grad: {param.requires_grad}")


        ################## 重みの読み込み
        if load_path:
            self.load_weight(load_path)

        ################## loss 関数の定義
        if self.train_subject == "goal_image_generator":
            self.loss = goal_image_loss()
            # self.loss = nn.L1Loss()
            # self.gradient_means = {name: [] for name, param in self.model.goal_image_generator.named_parameters()}
        elif self.train_subject == "manipulator_remover":
            ################################################ 関数作成
            self.loss = goal_image_loss()
        elif self.train_subject == "action_generator":
            self.loss = action_loss()
        elif self.train_subject == "all":
            self.loss = action_loss()

        ################## パラメータ設定
        self.epoch = epoch
        self.lerning_rate = lerning_rate
        self.batch_size = batch_size
        self.regularization = 0.03

        ################## optimazer の設定
        # print(self.model)
        if self.train_subject == "goal_image_generator":
            self.optimizer = optim.AdamW(           
                self.model.parameters(),
                lr=self.lerning_rate,
                weight_decay=self.regularization)
        elif self.train_subject == "manipulator_remover":
            self.optimizer = optim.AdamW(           
                self.model.parameters(),
                lr=self.lerning_rate,
                weight_decay=self.regularization)
        elif self.train_subject == "action_generator":
            self.optimizer = optim.AdamW(           
                self.model.parameters(),
                lr=self.lerning_rate,
                weight_decay=self.regularization)
        elif self.train_subject == "all":
            self.optimizer = optim.AdamW(           
                self.model.action_generator.parameters(),
                lr=self.lerning_rate,
                weight_decay=self.regularization)
        
        # for name, param in self.model.action_generator.named_parameters():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")

        self.dataset_base = "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/" 

        # self.check_dtype()
        self.model.to(torch.float32)

        self.transform = transforms.Resize((224, 224))
        self.transform_256 = transforms.Resize((256, 256))

    def forward(self, image1, image2 = None):
        image1 = image1.to(self.device)
        # image2 = image2.to(self.device)
        if self.train_subject == "goal_image_generator":
            self.x = self.model(image1)
        elif self.train_subject == "manipulator_remover":
            self.x = self.model(image1)
        elif self.train_subject == "action_generator":
            self.x = self.model(image1, image2)
        else:
            self.x = self.model(image1, image2)

        
        # return self.x

    def load_weight(self, path):
        weight = torch.load(path, map_location='cuda')
        weight = self.rename_key(weight)
        weight = self.delete_key(weight)
        # print()
        # print(weight.keys())
        self.model.load_state_dict(weight)
        self.model = self.model.to("cuda")

    def rename_key(self, d):
        keys_list = list(d.keys())
        # print(keys_list)
        for key in keys_list:
            new_key = key.replace('goal_image_generator.', '')
            d[new_key] = d.pop(key)
        return d
    
    def delete_key(self, weight):
        keys_to_delete = [key for key in weight.keys() if 'dooranglepredictor' in key]
        for key in keys_to_delete:
            del weight[key]


        return weight
    
    # def adapt_weight(self, weight):
    #     model_state_dict = self.model.state_dict()
    #     # state_dict = 
    #     for name, param in weight.items():
    #         if allowed_layers is None or any(layer in name for layer in allowed_layers):
    #             if name in model_state_dict:  # 層がモデルに存在するか確認
    #                 model_state_dict[name].copy_(param)
    #                 print(f"Loaded weights for layer: {name}")
    #             else:
    #                 print(f"Skipped layer: {name} (not in model)")
    #         else:
    #             print(f"Ignored layer: {name} (not in allowed layers)")

    
    def train_step(self, input1, output, epoch, input2 = None):
        # if epoch > 100:
        #     input1 = self.noiz(input1)
        input1 = input1.repeat([1, 3, 1, 1,])
        # input2 = input2.repeat([1, 3, 1, 1,])
        # print(input1.dtype)
        # print(output.dtype)
        self.model.train()
        self.optimizer.zero_grad()
        self.forward(input1, input2)
        # self.forward(input1.squeeze(), input2.squeeze())
        # print(f"y.grad_fn: {self.x.grad_fn}")
        loss = self.loss(self.x, output.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.model.goal_image_generator.named_parameters():
        #     if param.grad is None:
        #         print(f"{name} has no gradient. Check for detach or no_grad.")
        #     else:
        #         print(f"{name} gradient: {param.grad}")
        # print(input1.grad)
        # self.check_gradients(self.model.goal_image_generator)
        self.optimizer.step()
        ########################################################### 関数の設定
        accuracy = self.accuracy(self.x, output.to(self.device))

        # print(loss)

        return loss, accuracy

    def eval_step(self, input1, output, epoch, input2 = None):

        input1 = input1.repeat([1, 3, 1, 1,])
        # input2 = input2.repeat([1, 3, 1, 1,])
        # if epoch > 100:
        #     input1 = self.noiz(input1)
        
        self.model.eval()
        with torch.no_grad():
            self.forward(input1, input2)
            # self.forward(input1.squeeze(), input2.squeeze())
            loss = self.loss(self.x, output.to(self.device))
            ########################################################### 関数の設定
            accuracy = self.accuracy(self.x, output.to(self.device))

        return loss, accuracy
    
    def train_step_remover(self, input1, output, epoch,  input2 = None):
        # if epoch > 40:
        #     # noiz_level = (epoch-40) * 0.005
        #     input1 = self.add_noise(input1, noise_level)
        input1 = input1.repeat([1, 3, 1, 1,])
        self.model.train()
        self.optimizer.zero_grad()
        self.forward(input1, input2)
        # print(f"y.grad_fn: {self.x.grad_fn}")
        loss = self.loss(self.x, output.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.model.goal_image_generator.named_parameters():
        #     if param.grad is None:
        #         print(f"{name} has no gradient. Check for detach or no_grad.")
        #     else:
        #         print(f"{name} gradient: {param.grad}")
        # print(input1.grad)
        # self.check_gradients(self.model.goal_image_generator)
        self.optimizer.step()
        ########################################################### 関数の設定
        accuracy = self.accuracy(self.x, output.to(self.device))

        # print(loss)

        return loss, accuracy

    def eval_step_remover(self, input1, output, epoch,  input2 = None):
        # if epoch > 40:
        #     # noise_level = (epoch-40) * 0.005
        #     input1 = self.add_noise(input1, noise_level)
        input1 = input1.repeat([1, 3, 1, 1,])
        self.model.eval()
        with torch.no_grad():
            self.forward(input1, input2)
            loss = self.loss(self.x, output.to(self.device))
            ########################################################### 関数の設定
            accuracy = self.accuracy(self.x, output.to(self.device))

        return loss, accuracy
    
    def add_noise(self, data, std, mean = -0.2):
        noise = torch.randn(data.shape) * std + mean
        data += noise
        data = torch.where(data > 1, 1, data)
        data = torch.where(data < 0 , 1, data)
        return data

    def noiz(self, image_tensor, prob = 0.07):
        mean = 0.0
        std_dev = 0.1  # 標準偏差
        gaussian_noise = torch.randn_like(image_tensor) * std_dev + mean
        noisy_image_tensor = image_tensor + gaussian_noise
        noisy_image_tensor = torch.clamp(noisy_image_tensor, 0.0, 1.0)

        noise = torch.rand_like(image_tensor)
        salt_mask = noise < (prob / 2)  # 白（塩）になる部分
        noisy_image_tensor[salt_mask] = 1

        return noisy_image_tensor


    def make_dataset(self, current_image, target_image, action, batch = 4, eval = None):
        # print(self.dataset_base)
        # input
        self.eval = eval
        self.input_1 = torch.load(self.dataset_base + current_image).to(torch.float32)
        # self.input_1 = self.input_1.requires_grad()
        # self.input_1.requires_grad = True
        # print(f"grad ================= {self.input_1.requires_grad}")
        self.input_1 = self.input_1.view(-1, 1, 224, 224)
        # self.input_1 = self.transform_256(self.input_1)
        # self.input_1 = self.transform(self.input_1)
        # self.input_1 = self.input_1.to(torch.float32)

        if eval == True:
            self.save_id_eval = [345, 1023, 2800]
        # else:
        #     self.save_id_train = random.sample(range(self.input_1.shape[0]), 3) 
        
        if self.train_subject == 'goal_image_generator' or self.train_subject == "manipulator_remover":
            self.output = torch.load(self.dataset_base + target_image).to(torch.float32)
            # output = output[:2]
            self.output = self.transform_256(self.output.view(-1, 1, 224, 224))
            dataset = TensorDataset(self.input_1, self.output)
        else :
            self.input_2 = torch.load(self.dataset_base + target_image).to(torch.float32)
            self.input_2 = self.transform_256(self.input_2)
            output = torch.load(self.dataset_base + action).to(torch.float32)
            dataset = TensorDataset(self.input_1, self.input_2, output)

        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
        
        return dataloader

    def make_dataset_eval(self, current_image, target_image, action, batch = 4, eval = None):
        # print(self.dataset_base)
        # input
        self.eval = eval
        self.input_1_eval = torch.load(self.dataset_base + current_image).to(torch.float32)
        self.input_1_eval = self.input_1_eval.view(-1, 1, 224, 224)
        # self.input_1 = self.transform_256(self.input_1)
        # self.input_1_eval = self.transform(self.input_1_eval)
        self.save_id_eval = [345, 1023, 2800]
        
        if self.train_subject == 'goal_image_generator' or self.train_subject == "manipulator_remover":
            self.output_eval = torch.load(self.dataset_base + target_image).to(torch.float32)
            # output = output[:2]
            self.output_eval = self.transform_256(self.output_eval.view(-1, 1, 224, 224))
            dataset = TensorDataset(self.input_1_eval, self.output_eval)
        else :
            self.input_2_eval = torch.load(self.dataset_base + target_image).to(torch.float32)
            self.input_2 = self.transform_256(self.input_2)
            self.output_eval = torch.load(self.dataset_base + action).to(torch.float32)
            dataset = TensorDataset(self.input_1_eval, self.input_2_eval, self.output_eval)

        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
        
        return dataloader

    def modify_oprimazer(self, epoch):
        # if epoch < 120:
        if self.lerning_rate > 0.000001:
            self.lerning_rate /= 1.3
            self.optimizer = optim.AdamW(           
                self.model.parameters(),
                lr=self.lerning_rate,
                weight_decay=self.regularization) 
            
        # if epoch == 60:
        #     self.lerning_rate = 0.0001
        # if self.lerning_rate > 0.000001 and epoch > 59:
        #     self.lerning_rate /= 1.3
        #     self.optimizer = optim.AdamW(           
        #         self.model.manipulator_remover.parameters(),
        #         lr=self.lerning_rate,
        #         weight_decay=self.regularization) 
            
        # if self.lerning_rate > 0.000001:
        #     self.lerning_rate /= 1.3
        #     self.optimizer = optim.AdamW(           
        #         self.model.action_generator.parameters(),
        #         lr=self.lerning_rate,
        #         weight_decay=self.regularization) 

    def reset_optimizer(self):
        # if epoch < 120:
        self.lerning_rate = 0.0001
        self.optimizer = optim.AdamW(           
            self.model.parameters(),
            lr=self.lerning_rate,
            weight_decay=self.regularization) 

    def save_weight(self, path, epoch, last = None, best = None, noise = None):
        if last :
            save_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + path + f'/weight/weights_last.pth'
        elif best:
            save_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + path + '/weight/weights_best.pth'
        elif noise:
            save_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + path + f'/weight/weights_{noise}.pth'
        else:
            save_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + path + f'/weight/weights{epoch}.pth'
        
        torch.save(self.model.state_dict(), save_path) 

    def accuracy(self, predict, target):
        df  = torch.abs(predict[:, :3] - target[:, :3])
        # print(df.shape)
        numel = torch.numel(df)
        accuracy_1 = torch.where(df<0.01, 1, 0)
        accuracy_5 = torch.where(df<0.05, 1, 0)
        

        accuracy_1 = torch.sum(accuracy_1)
        accuracy_5 = torch.sum(accuracy_5)

        accuracy_1 =  float(accuracy_1) / float(numel)
        accuracy_5 =  float(accuracy_5) / float(numel)

        # accuracy_1 = ff.cosine_similarity(predict[:, :3], target[:, :3], 1, 1e-8).mean()

        return [accuracy_1, accuracy_5]

    def make_dir(self, path):
        directory_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + path
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            os.makedirs(directory_path + "/weight")
            os.makedirs(directory_path + "/loss")
            os.makedirs(directory_path + "/image")
            os.makedirs(directory_path + "/image/345")
            os.makedirs(directory_path + "/image/345/depth")
            os.makedirs(directory_path + "/image/345/input_image")
            # os.makedirs(directory_path + "/image/678")
            # os.makedirs(directory_path + "/image/678/depth")
            # os.makedirs(directory_path + "/image/678/input_image")
            os.makedirs(directory_path + "/image/1023")
            os.makedirs(directory_path + "/image/1023/depth")
            os.makedirs(directory_path + "/image/1023/input_image")
            os.makedirs(directory_path + "/image/2800")
            os.makedirs(directory_path + "/image/2800/depth")
            os.makedirs(directory_path + "/image/2800/input_image")

            # [345, 678, 1023, 1987]

        # for i in range()

    

    def save_sample_image(self, train_subject, path, epoch, eval = None):
        # print(path)
        directory_path = "/home/engawa/py_ws/visual_servo/src/network/sim2real/log/" + path + "/image"
        
        if eval:
            if epoch == 0:
                output_number = 0
                depth_set = self.transform_256(self.input_1[self.save_id_eval])
                for i in range(len(self.save_id_eval)):
                    depth = depth_set[i, 0].to('cpu').detach().numpy().copy()
                    
                    depth *= 255
                    depth = Image.fromarray((depth).astype(np.uint8))
                    depth.save(directory_path + f"/{self.save_id_eval[i]}/depth.png")
                    # print(data.shape)
                    # attension = self.input_1_eval[self.save_id_eval][i, 1].to('cpu').detach().numpy().copy()
                    # attension *= 255
                    # attension = Image.fromarray((attension).astype(np.uint8))
                    # # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
                    # attension.save(directory_path + f"/{self.save_id_eval[i]}/attension.png")

                    depth = self.output[self.save_id_eval][i, 0].to('cpu').detach().numpy().copy()
                    depth *= 255
                    depth = Image.fromarray((depth).astype(np.uint8))
                    depth.save(directory_path + f"/{self.save_id_eval[i]}/depth_goal.png")
                    # print(data.shape)
                    # attension = self.output_eval[self.save_id_eval][i, 1].to('cpu').detach().numpy().copy()
                    # attension *= 255
                    # attension = Image.fromarray((attension).astype(np.uint8))
                    # # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
                    # attension.save(directory_path + f"/{self.save_id_eval[i]}/attension_goal.png")
        

            with torch.no_grad():
                depth = self.input_1[self.save_id_eval]
                # if train_subject == "manipulator_remover":
                #         noiz_level = epoch * 0.01
                #         depth = self.add_noise(depth, noiz_level)
                depth = depth.repeat([1, 3, 1, 1,])
                self.forward(depth)
                for i in range(len(self.save_id_eval)):
                    depth_predict = self.x[i, 0].to('cpu').detach().numpy().copy()

                    depth_predict *= 255
                    depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
                    # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
                    depth_predict.save(directory_path + f"/{self.save_id_eval[i]}/depth/{epoch}.png")
                    input_depth = depth[i, 0].to('cpu').detach().numpy().copy()
                    input_depth *= 255
                    input_depth = Image.fromarray((input_depth).astype(np.uint8))
                    # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
                    input_depth.save(directory_path + f"/{self.save_id_eval[i]}/input_image/{epoch}.png")
            
            ###### 全データの保存
            # for i in range(self.input_1_eval.shape[0]):
            #         depth = depth_set[i, 0].to('cpu').detach().numpy().copy()
                    
            #         depth *= 255
            #         depth = Image.fromarray((depth).astype(np.uint8))
            #         depth.save(directory_path + f"/input/{i}.png")
            #         # print(data.shape)
            #         # attension = self.input_1_eval[self.save_id_eval][i, 1].to('cpu').detach().numpy().copy()
            #         # attension *= 255
            #         # attension = Image.fromarray((attension).astype(np.uint8))
            #         # # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            #         # attension.save(directory_path + f"/{self.save_id_eval[i]}/attension.png")

            #         depth = self.output[i, 0].to('cpu').detach().numpy().copy()
            #         depth *= 255
            #         depth = Image.fromarray((depth).astype(np.uint8))
            #         depth.save(directory_path + f"/output/{i}.png")

            #         # depth = self.output[i, 0].to('cpu').detach().numpy().copy()
            #         # depth *= 255
            #         # depth = Image.fromarray((depth).astype(np.uint8))
            #         # depth.save(directory_path + f"/predict/{i}.png")
            #         # print(data.shape)
            #         # attension = self.output_eval[self.save_id_eval][i, 1].to('cpu').detach().numpy().copy()
            #         # attension *= 255
            #         # attension = Image.fromarray((attension).astype(np.uint8))
            #         # # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            #         # attension.save(directory_path + f"/{self.save_id_eval[i]}/attension_goal.png")
        

            # with torch.no_grad():
            #     depth = self.input_1_eval
            #     # if train_subject == "manipulator_remover":
            #     #         noiz_level = epoch * 0.01
            #     #         depth = self.add_noise(depth, noiz_level)
            #     depth = depth.repeat([1, 3, 1, 1,])
            #     self.forward(depth)
            #     for i in range(self.input_1_eval.shape[0]):
            #         depth_predict = self.x[i, 0].to('cpu').detach().numpy().copy()

            #         depth_predict *= 255
            #         depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
            #         # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            #         depth_predict.save(directory_path + f"/depth/{i}.png")
            #         input_depth = depth[i, 0].to('cpu').detach().numpy().copy()
            #         input_depth *= 255
            #         input_depth = Image.fromarray((input_depth).astype(np.uint8))
            #         # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            #         input_depth.save(directory_path + f"/predict/{i}.png")


    def check_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.gradient_means[name].append(param.grad.abs().mean().item())
                print(f"Layer: {name}, Gradient norm: {param.grad.abs().mean()}")
            # else:
            #     print(f"Layer: {name} has no gradient")

    def plot_grad(self, epoch):
        for name, means in self.gradient_means.items():
            plt.plot(means, label=name)
            plt.xlabel('Training Step')
            plt.ylabel('Gradient Mean')
            plt.legend()
            plt.savefig(f'/home/engawa/py_ws/visual_servo/src/network/sim2real/check/{name}_{epoch}.png')
            plt.clf()


    def check_dtype(self):
        for name, param in self.model.named_parameters():
            print(f"Parameter: {name}, dtype: {param.dtype}")

    def forward_solo(self, input):
        input = self.transform(input)
        input = input.squeeze().unsqueeze(1)
        input = input.repeat([1, 3, 1, 1])
        # print(input.shape)
        self.forward(input)



        for i in range(self.x.shape[0]):
            depth = input[i, 0].to('cpu').detach().numpy().copy()
            depth *= 255
            depth = Image.fromarray((depth).astype(np.uint8))
            # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            depth.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/eval/goal_image/input/{i}.png")
            depth_predict = self.x[i, 0].to('cpu').detach().numpy().copy()
            depth_predict *= 255
            print(depth_predict.shape)
            depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
            # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            depth_predict.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/eval/goal_image/predict/{i}.png")
            # depth_predict = target[i, 0].to('cpu').detach().numpy().copy()
            # depth_predict *= 255
            # depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
            # # depth_predict.save(f"image/sice/{sample_id[i]}/{epoch}.png")
            # depth_predict.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/eval/remover/output/{i}.png")


    def add_noise_shapes(self, image):
        shape_number = random.randint(1, 4)
        # shape_number = 4
        std = random.uniform(0.01, 0.025)
        # std = 0.04
        image = self.add_noise(image, std)
        # end_center = [0, 0]
        for i in range(shape_number):
            noise_number = random.randint(0, 1)
            # if i == 0:
            #     noise_number = 0
            # else:
            #     noise_number = 1
            # noise_number = 1
            if noise_number == 0:
                image = self.add_noise_rectangle(image)
            elif noise_number == 1:
                image = self.add_noise_circle(image, i)

            save_image = image.to('cpu').detach().numpy().copy()

            save_image *= 255
            depth = Image.fromarray((save_image).astype(np.uint8))
            # depth.save(f"image/test_noiz/noiz_all_{i}.png")
        
        return image


    def add_noise_rectangle(self, image, mean=0.5, std_dev=0.01):
        count = True
        mean = random.uniform(0.2, 0.7)
        # mean = 0.55
        # print(mean)
        std_dev = random.uniform(0.01, 0.02)

        ## 形状の決定
        width_rect = random.randint(8, 40)
        height_rect = random.randint(40, 224)
        # height_rect = 224

        ###余白をつける
        width_rect += 8
        height_rect += 8
        if height_rect > 224:
            height_rect = 224

        

        x1 = random.randint(0, 224 - width_rect)
        x2 = x1 + width_rect
        y1 = random.randint(0, 224 - height_rect)
        y2 = y1 + height_rect

        rect_mask = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.float32)
        rect_mask[y1:y2, x1:x2] = torch.normal(mean=mean, std=std_dev, size=(height_rect, width_rect))
        ##edge にピクセル抜けを追加
        edge_number = random.randint(0, 2)

        # edge_number = 2
        if edge_number == 0 or edge_number == 2:
            noise_wide_1 = random.randint(1, int(width_rect/4) )
            rect_mask[y1:y2, x1:(x1 + noise_wide_1)] = torch.ones([y2 - y1, noise_wide_1])

        if edge_number == 1 or edge_number == 2:
            noise_wide_2 = random.randint(1, int(width_rect/8) )
            # print(noise_wide_2)
            if x2 + noise_wide_2 < 224:
                rect_mask[y1:y2, x2:x2 + noise_wide_2] = torch.ones([y2 - y1, noise_wide_2])

        angle = random.uniform(-60, 60)
        rect_mask = F.affine(rect_mask.unsqueeze(0), angle=angle, translate=(0, 0), scale=1.0, shear=0)[0]

        noisy_image = torch.where(rect_mask == 0, image, rect_mask)

        return noisy_image

    def point_slope_to_general(self, x, y, m):
        A = -m
        B = 1
        C = -(y - m * x)

        return A, B, C
    
    def add_noise_circle(self, im, i, noise_intensity=0.01):
        image = im.clone()
        mean = random.uniform(0.2, 0.7)
        # mean = 0.5
        noise_intensity = random.uniform(0.01, 0.02)
        # if end_center == [0, 0]:
        center = [random.randint(10, 213) for _ in range(2)]
        # else:
        #     center = end_center
        #     print(center)
        #     # center[0] -= 5
        #     # center[1] -= random.randint(-5, 5)
        radius = random.uniform(10., 25.)
        # radius = 25
        noiz_radius = radius + 2
        # noisy_image = image.clone()
        H, W = image.shape
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
        dist_noise = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
        zero_number = random.randint(0, 3)
        # print(zero_number)
        # zero_number = 0
        dist_noise -= torch.normal(mean=0, std=0.5, size=(H, W))
        if zero_number == 0:
            dist_noise[:int(center[1])] = 100
        elif zero_number == 1:
            dist_noise[int(center[1]):] = 100
        elif zero_number == 2:
            dist_noise[:, :int(center[0])] = 100
        elif zero_number == 3:
            dist_noise[:, int(center[0]):] = 100

        # dist_noise -= torch.normal(mean=-0.1, std=0.1, size=(H, W))


        mask = dist <= radius
        
        # print(torch.any(mask))
        mask_noise = dist_noise <= noiz_radius
        # print(f"center =========== {center}")
        zero = torch.ones_like(image)
        image = torch.where(mask_noise, zero, image)
        # save_image = image.to('cpu').detach().numpy().copy()
        # save_image *= 255
        # depth = Image.fromarray((save_image).astype(np.uint8))
        # depth.save(f"image/test_noiz/zero_test_{i}.png")

        noise = torch.normal(mean=mean, std=noise_intensity, size=(H, W))
        image = torch.where(mask, noise, image)

        # print("a")

        return image
    
# class train_function_2(train_function):
#     def __init__(self):
#         super.__init__(train_subject = "goal_image_generator", 
#                 load_path = None, epoch = 300000, lerning_rate = 0.005, batch_size = 16)
        
#     def add_noise_circle(self, )
