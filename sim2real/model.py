import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from collections import OrderedDict

import clip

# from pix2pix_network import define_G, define_D
# from gan_losses import GANLoss
# from loss import Loss

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class image_encoder(nn.Module):
    def __init__(self, input_resolution = 224, patch_size = 32, width=768, layers = 12, heads = 12, output_dim = 512):
        super(image_encoder, self).__init__()
        self.model, preprocess = clip.load("ViT-B/32", device="cuda")

    def forward(self, x: torch.Tensor):
        x = self.model.encode_image(x)

        return x

class image_decoder(nn.Module):
    def __init__(self):
        super(image_decoder, self).__init__()
        
        # 512次元の特徴量ベクトルから、最初の画像サイズに変換
        self.fc = nn.Linear(512, 8 * 8 * 128)  # 8x8サイズ、128チャネル
        
        # 逆畳み込み層でアップサンプリング
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()  
        )

    def forward(self, x):
        # 全結合層で形状変換
        x = x.to(torch.float32)
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)  # (バッチサイズ, チャネル, 高さ, 幅)
        
        # 逆畳み込み層でアップサンプリング
        x = self.deconv_layers(x)
        return x
    
class goal_image_generator(nn.Module):
    def __init__(self):
        super(goal_image_generator, self).__init__()
        # encoder = image_encoder()
        # decoder = image_decoder()
        self.encoder = image_encoder()
        self.decoder = image_decoder()
    
    def forward(self, image):
        # print("bbb")
        image = self.encoder(image)
        print(image.shape)
        image = self.decoder(image)
        
        return image
    
class manipulator_remover(nn.Module):
    def __init__(self):
        super(manipulator_remover, self).__init__()

        self.encoder = image_encoder()
        self.decoder = image_decoder()
    
    def forward(self, image):
        # print("bbb")
        image = self.encoder(image)
        image = self.decoder(image)
        
        return image
    
# class action_generator(nn.Module):
#     def __init__(self):
#         super(action_generator, self).__init__()
#         self.encoder = image_encoder()
#         self.decoder = action_decoder()
#         self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)


#     def forward(self,x1,x2):
#         x1 = self.encoder(x1)
#         x2 = self.encoder(x2)
#         x = x1 - x2
#         # print(x1)
#         # print(x2)
#         # print(x)
#         # print(x.shape)
#         # x = x.unsqueeze(1)
#         # x = self.Global_Avg_pooling(x)
#         # print(x.shape)
#         # x = x.to("torch.float32")
#         x = x.squeeze()
#         x = self.decoder(x)

#         return x
    
class action_decoder(nn.Module):
    def __init__(self):
        super(action_decoder, self).__init__()

        self.model = nn.Sequential(
                        nn.Linear(512, 6),
                        nn.Tanh()
        )

    def forward(self, x):
        # x = torch.tanh(self.model(x))
        x = self.model(x)

        return x

class ivis(nn.Module):
    def __init__(self):
        super(ivis, self).__init__()
        self.goal_image_generator = goal_image_generator()

        ####################　実装する
        self.manipulator_remover = manipulator_remover()
        self.action_generator = action_generator()

    def forward(self, image):
        goal_image = self.goal_image_generator(image)

        return goal_image
    
class action_generator(nn.Module):
    def __init__(self):
        super(action_generator, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder(6)
        self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = x1 - x2
        x = self.Global_Avg_pooling(x)
        x = x.squeeze()
        x = self.decoder(x)

        return x
    
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
                        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        x = self.model(x)

        return x

class decoder(nn.Module):
    def __init__(self, output):
        super(decoder, self).__init__()

        self.model = nn.Sequential(
                        nn.Linear(512, output),
                        nn.Tanh()
        )

    def forward(self, x):
        # x = torch.tanh(self.model(x))
        x = self.model(x)

        return x


if __name__ == "__main__": 
    model = image_encoder()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# import numpy as np
# from collections import OrderedDict
# import torchvision.transforms as T

# import clip

# # from pix2pix_network import define_G, define_D
# # from gan_losses import GANLoss
# # from loss import Loss

# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)

# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x

# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

# class image_encoder(nn.Module):
#     def __init__(self, input_resolution = 224, patch_size = 32, width=768, layers = 12, heads = 12, output_dim = 512):
#         super(image_encoder, self).__init__()
#         self.model, preprocess = clip.load("ViT-B/32", device="cuda")
#         self.preprocess = T.Compose([
#                             T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                 std=(0.26862954, 0.26130258, 0.27577711))
# ])
#         # print(preprocess)

#     def forward(self, x: torch.Tensor):
#         # print(self.preprocess)
#         x = self.preprocess(x)
#         x = self.model.encode_image(x)

#         return x

# class image_decoder(nn.Module):
#     def __init__(self):
#         super(image_decoder, self).__init__()
        
#         # 512次元の特徴量ベクトルから、最初の画像サイズに変換
#         self.fc = nn.Linear(512, 8 * 8 * 128)  # 8x8サイズ、128チャネル
        
#         # 逆畳み込み層でアップサンプリング
#         self.deconv_layers = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1), 
#             nn.Tanh()  
#         )

#     def forward(self, x):
#         # 全結合層で形状変換
#         x = x.to(torch.float32)
#         x = self.fc(x)
#         x = x.view(-1, 128, 8, 8)  # (バッチサイズ, チャネル, 高さ, 幅)
        
#         # 逆畳み込み層でアップサンプリング
#         x = self.deconv_layers(x)
#         return x
    
# class image_decoder_trick(nn.Module):
#     def __init__(self):
#         super(image_decoder_trick, self).__init__()
        
#         # 512次元の特徴量ベクトルから、最初の画像サイズに変換
#         self.fc = nn.Linear(513, 513 * 8 * 8)  # 8x8サイズ、128チャネル
        
#         # 逆畳み込み層でアップサンプリング
#         self.deconv_layers = nn.Sequential(
#             nn.ConvTranspose2d(513, 256, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), 
#             nn.Tanh()  
#         )

#     def forward(self, x):
#         # 全結合層で形状変換
#         x = x.to(torch.float32)
#         x = self.fc(x)
#         x = x.view(-1, 513, 8, 8)  # (バッチサイズ, チャネル, 高さ, 幅)args.target_path_eval,
        
#         # 逆畳み込み層でアップサンプリング
#         x = self.deconv_layers(x)
#         return x
    
# class TransformerBlock(nn.Module):
#     """
#     簡易版の Transformer デコーダブロック (自己注意 + MLP)
#     """
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio

#         # LayerNorm
#         self.ln1 = nn.LayerNorm(embed_dim)
#         self.ln2 = nn.LayerNorm(embed_dim)

#         # Multi-Head Self-Attention
#         self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

#         # MLP
#         hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         # x: (B, N, embed_dim)
#         # Self-Attention (残差接続 + LayerNorm)
#         x_norm = self.ln1(x)
#         attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
#         x = x + attn_out

#         # MLP (残差接続 + LayerNorm)
#         x_norm = self.ln2(x)
#         mlp_out = self.mlp(x_norm)
#         x = x + mlp_out

#         return x
    
# class ViTDecoder(nn.Module):
#     """
#     513次元の特徴ベクトル → 1×256×256 画像を生成する ViT デコーダの一例
#     """
#     def __init__(
#         self,
#         latent_dim=513,     # ViTエンコーダ出力の次元
#         embed_dim=512,      # デコーダ内部の埋め込み次元
#         num_tokens=256,     # 出力したいパッチの数 (16x16=256)
#         patch_size=16,      # パッチのピクセル数 (16x16)
#         num_layers=4,       # Transformer ブロック数
#         num_heads=8,        # マルチヘッドアテンションのヘッド数
#         mlp_ratio=4.0,
#         dropout=0.1
#     ):
#         super().__init__()

#         self.latent_dim = latent_dim
#         self.embed_dim = embed_dim
#         self.num_tokens = num_tokens
#         self.patch_size = patch_size
#         self.num_layers = num_layers

#         # 1) Latent embedding: 513次元 → embed_dim
#         self.latent_to_embed = nn.Linear(latent_dim, embed_dim)

#         # 2) Decoder Tokens: 学習可能なトークン (num_tokens 個)
#         self.decoder_tokens = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

#         # 3) Positional Embedding (num_tokens 個分)
#         self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

#         # 4) Transformer Blocks (自己注意のみの簡易デコーダ)
#         self.blocks = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
#             for _ in range(num_layers)
#         ])

#         # 最終出力: パッチ (16×16=256 画素) を予測
#         self.fc_out = nn.Linear(embed_dim, patch_size*patch_size)

#     def forward(self, z):
#         """
#         z: (B, 513) - エンコーダからの特徴ベクトル
#         出力: (B, 1, 256, 256) - グレースケール画像
#         """
#         B = z.size(0)

#         # 1) Latent埋め込みへ
#         latent_emb = self.latent_to_embed(z)  # (B, embed_dim)

#         # 2) decoder_tokens を バッチ分複製
#         #    shape: (1, num_tokens, embed_dim) → (B, num_tokens, embed_dim)
#         dec_tokens = self.decoder_tokens.repeat(B, 1, 1)  # (B, N, embed_dim)
#         pos_emb = self.pos_embed.repeat(B, 1, 1)          # (B, N, embed_dim)

#         # 3) “latent_emb” を各 token へ注入する方法はいくつかあります。
#         #    ここでは簡易的に token の 0番目 に latent_emb を加算するなどの例を示します。
#         #    実際には cross-attention を用いる、あるいは tokens にconcat する等の設計もあり得ます。
        
#         # 例: 0番目のトークンに latent_emb を加算 (簡易版)
#         dec_tokens[:, 0, :] = dec_tokens[:, 0, :] + latent_emb

#         # 4) Positional Embedding を加算
#         x = dec_tokens + pos_emb  # (B, N, embed_dim)

#         # 5) Transformer デコーダ処理
#         for blk in self.blocks:
#             x = blk(x)  # (B, N, embed_dim)

#         # 6) パッチ出力
#         #    各トークンを patch_size*patch_size に写像する
#         #    shape: (B, N, patch_size^2)
#         patches = self.fc_out(x)  # (B, N, 256)

#         # 7) それぞれのトークンを 16×16 のパッチ画像に reshape し、全体で 256×256 画像を作る
#         #    N = 16x16=256 個のトークン → 各トークンは (16×16) ピクセル
#         #    順番に敷き詰める必要があるが、ここでは単純に (B, 256, 16, 16) に reshape し
#         #    行列的に 16×16 グリッドに並べて 256×256 にする例を示す
#         patches = patches.view(B, self.num_tokens, self.patch_size, self.patch_size)  # (B, 256, 16, 16)

#         # パッチを最終的に 1枚にタイル状に並べるために、一旦 (B, 16, 16, 16, 16) へ変形し
#         # dimensionを入れ替えて (B, 1, 256, 256) を作る。
#         # パッチの配置を正しく行うには、「トークンを 16x16 に並べる」ためのリシェイプが必要
#         # ここでは単純に行方向・列方向が [16, 16] 個のパッチと仮定。
#         patches = patches.view(B, 16, 16, self.patch_size, self.patch_size)
#         # (B, 16, 16, 16, 16) → (B, 1, 256, 256)
#         #  1) 16,16 をそれぞれ縦方向・横方向に連結
#         out = patches.permute(0, 1, 3, 2, 4).contiguous()
#         #  out shape: (B, 16, 16, 16, 16) -> (B, 16, 16, 16, 16) (permuteして軸入れ替え)
#         #  最終的に (B, 16*16, 16*16) = (B, 256, 256) へリシェイプ
#         out = out.view(B, 1, 256, 256)

#         out = torch.where(out > 1. , 1. , out)

#         return out
    
# class goal_image_generator(nn.Module):
#     def __init__(self):
#         super(goal_image_generator, self).__init__()
#         # encoder = image_encoder()
#         # decoder = image_decoder()
#         self.encoder = image_encoder()
#         self.decoder = image_decoder_trick()
#         self.decoder = ViTDecoder()
    
#     def forward(self, image, rot):
#         # print("bbb")
#         image = self.encoder(image)
#         # print(image.shape)
#         # print(rot.shape)
#         # print(image.shape)

#         image = torch.cat([image, rot], dim = 1)
#         image = self.decoder(image)
        
#         return image
    
# class manipulator_remover(nn.Module):
#     def __init__(self):
#         super(manipulator_remover, self).__init__()

#         self.encoder = image_encoder()
#         self.decoder = image_decoder()
    
#     def forward(self, image):
#         # print("bbb")
#         image = self.encoder(image)
#         image = self.decoder(image)
        
#         return image
    
# # class action_generator(nn.Module):
# #     def __init__(self):
# #         super(action_generator, self).__init__()
# #         self.encoder = image_encoder()
# #         self.decoder = action_decoder()
# #         self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)


# #     def forward(self,x1,x2):
# #         x1 = self.encoder(x1)
# #         x2 = self.encoder(x2)
# #         x = x1 - x2
# #         # print(x1)
# #         # print(x2)
# #         # print(x)
# #         # print(x.shape)
# #         # x = x.unsqueeze(1)
# #         # x = self.Global_Avg_pooling(x)
# #         # print(x.shape)
# #         # x = x.to("torch.float32")
# #         x = x.squeeze()
# #         x = self.decoder(x)

# #         return x
    
# class action_decoder(nn.Module):
#     def __init__(self):
#         super(action_decoder, self).__init__()

#         self.model = nn.Sequential(
#                         nn.Linear(512, 6),
#                         nn.Tanh()
#         )

#     def forward(self, x):
#         # x = torch.tanh(self.model(x))
#         x = self.model(x)

#         return x

# class ivis(nn.Module):
#     def __init__(self):
#         super(ivis, self).__init__()
#         self.goal_image_generator = goal_image_generator()

#         ####################　実装する
#         self.manipulator_remover = manipulator_remover()
#         self.action_generator = action_generator()

#     def forward(self, image):
#         goal_image = self.goal_image_generator(image)

#         return goal_image
    
# class action_generator(nn.Module):
#     def __init__(self):
#         super(action_generator, self).__init__()
#         self.encoder = encoder()
#         self.decoder = decoder(6)
#         self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)

#     def forward(self, x1, x2):
#         x1 = self.encoder(x1)
#         x2 = self.encoder(x2)
#         x = x1 - x2
#         x = self.Global_Avg_pooling(x)
#         x = x.squeeze()
#         x = self.decoder(x)

#         return x
    
# class encoder(nn.Module):
#     def __init__(self):
#         super(encoder, self).__init__()
#         self.model = nn.Sequential(
#                         nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(64),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(64),
#                         nn.ReLU(),
#                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#                         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(),
#                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#                         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(256),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(256),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(256),
#                         nn.ReLU(),
#                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
#                         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), 
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(),
#                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(),
#                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )

#     def forward(self, x):
#         x = self.model(x)

#         return x

# class decoder(nn.Module):
#     def __init__(self, output):
#         super(decoder, self).__init__()

#         self.model = nn.Sequential(
#                         nn.Linear(512, output),
#                         nn.Tanh()
#         )

#     def forward(self, x):
#         # x = torch.tanh(self.model(x))
#         x = self.model(x)

#         return x


# if __name__ == "__main__": 
#     model = image_encoder()