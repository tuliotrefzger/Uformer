import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import torchvision
from skimage.util import random_noise

from model import Uformer

#################################### UTILS ###############################################


def remove_files_from_dir(dir):
    for directory_path, _, files in os.walk(dir):
        print("Directory path:", directory_path)
        for file in files:
            file = directory_path + "/" + file
            print("File:", file)
            if os.path.isfile(file):
                print("Deleting file:", file)
                os.remove(file)
        print()


def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return img


# deg_level de 1 a 10
def generate_ruido_gaussiano(img, deg_level):
    # Noise Levels
    sigmas = np.linspace(1, 10, 10) / 40
    return to_0255(random_noise(img, var=sigmas[deg_level - 1] ** 2))


###################################### START #############################################


print("START")

directory = "patientImages/splits"
target_directory = "patientImages/splitsGaussianNoise"

for directory_path, subdirectories, files in os.walk(directory):
    for file in files:
        if file.lower().endswith(".png"):
            file = directory_path + "/" + file
            img = torch.from_numpy(np.float32(load_img(file)))
            print("image shape:", img.shape)

print("END")


# for filename in os.listdir(target_directory):
#     print(filename)
#     file = target_directory + "/" + filename
#     print(file)
#     if os.path.isfile(file):
#         print("Deleting file:", file)
#         os.remove(file)

# for directory_path, subdirectories, files in os.walk(directory):
#     print("Current Path: ", directory_path)
#     print("Directories: ", subdirectories)
#     print("Files: ", files)
#     print()
