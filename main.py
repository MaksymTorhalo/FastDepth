# import dataloader
import model
import train
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
import os
import sys




save_path = "./models"
temp_path = "./temp"        # directory for intermediate model states
sample_path = "./Samples"
os.makedirs(sample_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)


# img = Image.open("Samples/20241229182840_1.jpg").convert("RGB")
# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# input = transform(img).unsqueeze(0)
#
# fastDepth = model.NNConv5()
# x = torch.randn(1, 3, 224, 224)
# out = fastDepth(input).detach().squeeze().numpy()
#
# plt.imshow(out)
# plt.show()


train_flag = False
resume_training = False
while True:
    initial_prompt = input("Load or Train the Model? (l/t): ").strip().lower()
    if initial_prompt == "l":
        name = input("Please enter the model save file name:")
        print(f'Loading model {name} from {save_path}')
        state_dict = torch.load(f"{save_path}/{name}", weights_only=True)

        fastDepth = model.NNConv5()
        fastDepth.load_state_dict(state_dict)
        fastDepth.eval()

        while True:
            eval_input = input("Please enter the full file name of the Sample for evaluation or (e) for exit:")
            if eval_input == "e":
                break
            else:
                if os.path.exists(f"{sample_path}/{eval_input}"):
                    img = Image.open(f"{sample_path}/{eval_input}").convert("RGB")
                    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    input = transform(img).unsqueeze(0)

                    out = fastDepth(input).detach().squeeze().numpy()
                    plt.imshow(out, cmap="plasma")
                    plt.colorbar(label="Depth [m]")
                    plt.title(f'Prediction for {eval_input}')
                    plt.show()
                else:
                    print("Invalid file name. File should be located in the Samples directory.")
        break
    elif initial_prompt == "t":
        train_flag = True
        if os.path.exists(f'{temp_path}/FastDepth_temp.pth'):
            while True:
                prompt = input("Checkpoint was found. Resume model training from checkpoint (y/n)? ")
                if prompt == "y":
                    resume_training = True
                    break
                elif prompt == "n":
                    print("Checkpoint ignored. Starting training from initial state.")
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        break
    else:
        print("Invalid input. Please enter 'l' for Load or 't' for Train.")


if train_flag:
    train.train(20, resume_training=False)