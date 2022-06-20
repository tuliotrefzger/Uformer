from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross
import torch

loaded_model = Uformer()

FILE = "uformer16_denoising_sidd.pth"

loaded_model.load_state_dict(torch.load(FILE))

# loaded_model = torch.load(FILE)
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)
