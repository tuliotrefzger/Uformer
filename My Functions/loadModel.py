from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross, Downsample, Upsample
from collections import OrderedDict

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def return2originalSize(square_img, original_height, original_width):
    _, _, square_size, _ = square_img.size()
    #     print("square_size: ", square_size)
    horizontal_fill = square_size - original_width
    #     print("horizontal_fill: ", horizontal_fill)
    vertical_fill = square_size - original_height
    #     print("vertical_fill: ", vertical_fill)

    if horizontal_fill % 2:
        padding_left = int((horizontal_fill - 1) / 2)
        padding_right = int((horizontal_fill + 1) / 2)
    else:
        padding_left = int(horizontal_fill / 2)
        padding_right = padding_left

    if vertical_fill % 2:
        padding_top = int((vertical_fill - 1) / 2)
        padding_bottom = int((vertical_fill + 1) / 2)
    else:
        padding_top = int(vertical_fill / 2)
        padding_bottom = padding_top

    #     print("padding_left: ", padding_left)
    #     print("padding_right: ", padding_right)
    #     print("padding_top: ", padding_top)
    #     print("padding_bottom: ", padding_bottom)

    return square_img[
        :,
        :,
        padding_top : (original_height + padding_top),
        padding_left : (original_width + padding_left),
    ]


##########################################################################################
print("BEGINNING")
model = Uformer(embed_dim=16, token_mlp="leff", img_size=128, use_checkpoint=True)
model = torch.nn.DataParallel(model)

FILE = "uformer16_denoising_sidd.pth"

model.load_state_dict(torch.load(FILE)["state_dict"])

clean = torch.from_numpy(np.float32(load_img("GT_SRGB_010.PNG")))
noisy = torch.from_numpy(np.float32(load_img("NOISY_SRGB_010.PNG")))

original_height, original_width, _ = clean.shape
# print("original height: ", original_height)
# print("original width: ", original_width)
# print("clean Dimensions: ", clean.shape)

clean = clean.permute(2, 0, 1)
noisy = noisy.permute(2, 0, 1)

# print("Modified Noisy Dimensions 1: ", noisy.shape)

# noisy = torchvision.transforms.Resize(256)(noisy)
# clean = torchvision.transforms.Resize(256)(clean)

# noisy = torchvision.transforms.Resize(1000)(noisy)
# clean = torchvision.transforms.Resize(1000)(clean)

_, original_height, original_width = clean.shape
# print("height: ", original_height)
# print("width: ", original_width)
# print("clean Dimensions: ", clean.shape)


# print("Modified Noisy Dimensions 2: ", noisy.shape)

torch.cuda.empty_cache()
model.cuda()
model.eval()

noisy = noisy.cuda()
noisy = noisy.unsqueeze(0)
noisy, mask = expand2square(noisy, factor=128)
# print("Square shape: ", noisy.shape)
torch.cuda.empty_cache()
restored = model(noisy)
cv2.imwrite(
    "./SQUARED_SRGB_010.png",
    restored.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255,
)
# restored = restored.squeeze(0).detach().cpu().permute(1,2,0).numpy()

restored = return2originalSize(restored, original_height, original_width)
# print("Restored shape: ", restored.shape)
restored *= 255

cv2.imwrite(
    "./RESTORED_SRGB_010.png",
    restored.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(),
)
print("END")
