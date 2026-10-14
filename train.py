import dataloader
import model as md
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils import eval_metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# for daemon thread that saves the intermediate model state
import threading
import os
import sys
import time




cudnn.benchmark = True      # let CUDA choose the optimal algorithm for loading data into memory
save_path = "./models"
temp_path = "./temp"        # directory for intermediate model states
tensorboard_path = "./tensorboard"
writer = SummaryWriter(tensorboard_path)


freeze_request = threading.Event()
exit_request = threading.Event()

def command_listener():
    for line in sys.stdin:
        cmd = line.strip().lower()

        if cmd == "/exit":
            exit_request.set()
            break

        elif cmd == "/freeze":
            freeze_request.set()
            break


def train(epochs, resume = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = md.NNConv5().to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    start_epoch = 0
    best_rmse = float("inf")


    if resume and os.path.exists(f"{temp_path}/FastDepth_temp.pth"):
        checkpoint = torch.load(f"{temp_path}/FastDepth_temp.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_rmse = checkpoint['best_rmse']

        print(f'Intermediate model state loaded. Resuming training from epoch {start_epoch}.')
    else:
        print(f'Intermediate model state not found. Training from scratch.')


    listener_thread = threading.Thread(target=command_listener, daemon=True)
    listener_thread.start()


    if not resume and start_epoch < 3:
        for param in model.encoder.parameters():
            param.requires_grad = False
            ''' ensure that the pretrained encoder is not 
            being fitted to a degenerate decoder during
            initial epochs '''

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch} started ---------------')
        model.train()
        total_loss = 0.0

        if epoch == 3:
            for param in model.encoder.parameters():
                param.requires_grad = True

        for idx, batch in enumerate(dataloader.train_loader):
            images = batch["image"].to(device)
            gts = batch["gt"].to(device)

            optimizer.zero_grad()


            preds = model(images)
            valid_mask = (gts > 0).detach()     # bool tensor mask for gts that contain false data

            if valid_mask.sum() > 0:            # filter out invalid batches
                loss = criterion(preds[valid_mask], gts[valid_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if idx % 10 == 0:
                print(f'Batch {idx}/{len(dataloader.train_loader)} | Train Loss: {loss.item()}')
                writer.add_scalar('Loss/train', loss.item() , idx + len(dataloader.train_loader) * epoch)

            del images, gts, preds, loss    # free up memory before garbage collection, results in lower temporal RAM/VRAM usage


            if freeze_request.is_set():
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_rmse': best_rmse}, f'{temp_path}/FastDepth_temp.pth')
                print("Intermediate model state saved. Exiting in 3s...")
                time.sleep(3)
                sys.exit(0)
            elif exit_request.is_set():
                now = datetime.now()
                t_now = now.strftime("%Y_%m_%d_%H_%M")

                torch.save(model.state_dict(), f'{save_path}/FastDepth_{t_now}.pth')
                print(f'Model FastDepth_{t_now}.pth saved at {save_path}. Exiting in 3s...')
                time.sleep(3)
                sys.exit(0)

        avg_loss = total_loss / len(dataloader.train_loader)
        print(f'Epoch Train Loss: {avg_loss} ---------------')


        # Evaluation stage
        model.eval()
        val_rmse = 0.0
        val_delta1 = 0.0

        with torch.no_grad():
            for idx, batch in enumerate(dataloader.validation_loader):
                images = batch["image"].to(device)
                gts = batch["gt"].to(device)

                preds = model(images)

                rmse, delta1 = eval_metrics(preds, gts)
                val_rmse += rmse
                val_delta1 += delta1

                del images, gts, preds  # free up memory before garbage collection, results in lower temporal RAM/VRAM usage

        avg_val_rmse = val_rmse / len(dataloader.validation_loader)
        avg_val_delta1 = val_delta1 / len(dataloader.validation_loader)

        print(f'Epoch Validation RMSE: {avg_val_rmse} | Validation Delta 1: {avg_val_delta1}')
        writer.add_scalar('RMSE/eval', avg_val_rmse, epoch)
        writer.add_scalar('Delta1/eval', avg_val_delta1, epoch)

        if avg_val_rmse < best_rmse:
            best_rmse = avg_val_rmse

            now = datetime.now()
            t_now = now.strftime("%Y_%m_%d_%H_%M")

            torch.save(model.state_dict(), f'{save_path}/FastDepth_{t_now}.pth')
            print(f'Model FastDepth_{t_now}.pth saved at {save_path}.')