import dataloader
import model as md
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils import eval_metrics
from datetime import datetime




cudnn.benchmark = True
save_path = "./models"


def train(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = md.NNConv5().to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    best_RMSE = float("inf")

    for epoch in range(epochs):
        print(f'Epoch {epoch} started ---------------')
        model.train()
        total_loss = 0.0

        for idx, batch in enumerate(dataloader.train_loader):
            images = batch["image"].to(device)
            gts = targets["gt"].to(device)

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

        avg_loss = total_loss / len(dataloader.train_loader)
        print(f'Epoch Train Loss: {avg_loss} ---------------')


        # Evaluation stage
        model.eval()
        val_rmse = 0.0
        val_delta1 = 0.0

        with torch.no_grad():
            for idx, batch in enumerate(dataloader.validation_loader):
                images = batch["image"].to(device)
                gts = targets["gt"].to(device)

                preds = model(images)

                rmse, delta1 = eval_metrics(preds, gts)
                val_rmse += rmse
                val_delta1 += delta1

        avg_val_rmse = val_rmse / len(dataloader.validation_loader)
        avg_val_delta1 = val_delta1 / len(dataloader.validation_loader)

        print(f'Epoch Validation RMSE: {avg_val_rmse} | Validation Delta 1: {avg_val_delta1}')

        if avg_val_rmse < best_RMSE:
            best_RMSE = avg_val_rmse

            now = datetime.now()
            t_now = now.strftime("%Y_%m_%d_%H_%M")

            torch.save(model.state_dict(), f'{save_path}/FastDepth_{t_now}.pth')
            print(f'Model FastDepth_{t_now}.pth saved at {save_path}.')