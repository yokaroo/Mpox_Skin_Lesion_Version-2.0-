import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils.getData import getImageLabel
from torch.optim import Adam
from models.cnn import SimpleCNN


def main():
    BATCH_SIZE = 32
    EPOCH = 15
    LEARNING_RATE = 0.001
    folds = [1, 2, 3, 4, 5]
    DEVICE = 'cpu'

    train_aug_loader = DataLoader(
        getImageLabel(augmented=f'./dataset/Augmented Images/Augmented Images/FOLDS_AUG/', folds=folds,
                      subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    train_ori_loader = DataLoader(
        getImageLabel(original=f'./dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Train']),
        batch_size=BATCH_SIZE, shuffle=True)
    vali_loader = DataLoader(
        getImageLabel(original=f'./dataset/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Valid']),
        batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(input_dim=32, input_c=3, output=6, device=DEVICE)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_vali_all = [], []
    for epoch in range(EPOCH):
        train_loss = 0
        vali_loss = 0
        model.train()
        for batch, (src, trg) in enumerate(train_aug_loader):
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        for batch, (src, trg) in enumerate(train_ori_loader):
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        model.eval()
        for batch, (src, trg) in enumerate(vali_loader):
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            loss = loss_function(pred, trg)
            vali_loss += loss.detach().numpy()

        loss_train_all.append(train_loss / (len(train_aug_loader) + len(train_ori_loader)))
        loss_vali_all.append(vali_loss / len(vali_loader))
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / (len(train_aug_loader) + len(train_ori_loader))}, Validation Loss: {vali_loss / len(vali_loader)}')

        if (epoch + 1) % 15 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (len(train_aug_loader) + len(train_ori_loader)),
            }, "./SimpleCNN_" + str(epoch + 1) + ".pt")

    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_vali_all, color="#3399e6", label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./training.png")


if __name__ == "__main__":
    main()