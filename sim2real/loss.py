import torch
from torch import nn
import torch.nn.functional as F

class goal_image_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, output, target):

        loss_1 = self.loss(output, target)
        # print(torch.isnan(output).any())
        # print(torch.isnan(target).any())
        # print(torch.abs(output - target).shape)
        # print(torch.abs((output[:, 0] - target[:, 0]) * target[:, 1]).shape)
        # loss_2 = torch.mean(torch.abs(3*(output[:, 0] - target[:, 0]) * target[:, 1]))
        # loss_3 = torch.mean(torch.abs(1000*(output[:, 0] - target[:, 0]) * target[:, 1]))

        # loss_3 =  torch.mean(torch.abs(10*(output[:, 0] - target[:, 0]) * target[:, 1]) * torch.exp((torch.abs(output[:, 1] - target[:, 1]))*0.69) - 1)

        # print(target)

        # print(f"1 ================= {torch.abs(3*(output[:, 0] - target[:, 0]) * target[:, 1])}")
        # print(f"2 ================= {torch.exp((torch.abs(output[:, 1] - target[:, 1]))*0.69) - 1} ")

        # return loss_1 + loss_3

        return loss_1

class action_loss(nn.Module):
    def __init__(self): # パラメータの設定など初期化処理を行う
        # super(Loss, self).__init__()
        super(action_loss, self).__init__()
        self.list_p = [0, 1, 2]
        self.list_r = [3, 4, 5]
        # self.e = self.get_e()
        self.dim = 1
        self.eps = 1e-8

    def forward(self, outputs, targets):
        # print(outputs.shape)
        # print(targets.shape)
        
        de_p = torch.mean(torch.sqrt(torch.sum((outputs[:, :3] - targets[:, :3])**2, dim = 1)))
        de_r = torch.mean(torch.sqrt(torch.sum((outputs[:, 3:] - targets[:, 3:])**2, dim = 1)))
        position_cos_similarity = - F.cosine_similarity(outputs[:, :3], targets[:, :3], self.dim, self.eps).mean()
        rotation_cos_similarity = - F.cosine_similarity(outputs[:, 3:], targets[:, 3:], self.dim, self.eps).mean()

        return de_p + position_cos_similarity + de_r + rotation_cos_similarity


