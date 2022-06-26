from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross, Downsample, Upsample
from collections import OrderedDict

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import torchvision

# Utils
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return img


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[
        :, :, ((X - h) // 2) : ((X - h) // 2 + h), ((X - w) // 2) : ((X - w) // 2 + w)
    ] = timg
    mask[
        :, :, ((X - h) // 2) : ((X - h) // 2 + h), ((X - w) // 2) : ((X - w) // 2 + w)
    ].fill_(1.0)

    return img, mask


model = Uformer(embed_dim=16, token_mlp="leff")
model = torch.nn.DataParallel(model)

FILE = "uformer16_denoising_sidd.pth"

model.load_state_dict(torch.load(FILE)["state_dict"])

model.eval()

# Test

clean = torch.from_numpy(np.float32(load_img("GT_SRGB_010.PNG")))
noisy = torch.from_numpy(np.float32(load_img("NOISY_SRGB_010.PNG")))
clean = clean.permute(2, 0, 1)
noisy = noisy.permute(2, 0, 1)
noisy = torchvision.transforms.Resize(256)(noisy)
clean = torchvision.transforms.Resize(256)(clean)

plt.imshow(clean.permute(1, 2, 0))
plt.figure()
plt.imshow(noisy.permute(1, 2, 0))

torch.cuda.empty_cache()
# model.cuda()
model.eval()

noisy = noisy.cuda()
noisy = noisy.unsqueeze(0)
noisy, mask = expand2square(noisy, factor=128)
plt.figure()
plt.imshow(noisy.cpu().squeeze(0).permute(1, 2, 0))
print(noisy.shape)
torch.cuda.empty_cache()
restored = model(noisy)
plt.figure()
restored = restored.squeeze(0).detach()
plt.imshow(restored.cpu().permute(1, 2, 0))

# for param in model.parameters():
#     print(param)


# loaded_model = Uformer(img_size=128, in_chans=3,
#                  embed_dim=16, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
#                  win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, patch_norm=True,
#                  use_checkpoint=False, token_projection='linear', token_mlp='', se_layer=False,
#                  dowsample=Downsample, upsample=Upsample)

# FILE = "uformer16_denoising_sidd.pth"

# loaded_checkpoint = torch.load(FILE)
# print(loaded_checkpoint)

# epoch = loaded_checkpoint["epoch"]
# model = Uformer(embed_dim=16)
# model.load_state_dict(loaded_checkpoint["state_dict"])

# model.eval()

# for param in model.parameters():
#     print(param)


# loaded_model.load_state_dict(loaded_checkpoint)

# loaded_model = torch.load(FILE)
# loaded_model.eval()

# for param in loaded_model.parameters():
#     print(param)
