import os
import json
import torch
from utils.utils import get_classes
from utils.transforms import Normalize, ToTensor, RandRotation_z, RandomNoise
from torchvision import transforms, utils
from models.pointnet import PointNetCls, feature_transform_regularizer
from dataloader import MechanicalData
from torch.utils.data import DataLoader


def default_transforms():
    return transforms.Compose([Normalize(), ToTensor()])


def train_transforms():
    return transforms.Compose(
        [Normalize(), RandRotation_z(), RandomNoise(), ToTensor()]
    )


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (
        torch.norm(diff3x3) + torch.norm(diff64x64)
    ) / float(bs)


dataset_path = r"C:\Users\LukasPilsl\source\studium\bachelor\mechanicalpc\data"
train_annot = {}
with open(os.path.join(dataset_path, "annot_train.json")) as f:
    train_annot = json.load(f)
val_annot = {}
with open(os.path.join(dataset_path, "annot_test.json")) as f:
    val_annot = json.load(f)
classes = get_classes(train_annot)


train_ds = MechanicalData(dataset_path, train_annot, classes, train_transforms())
val_ds = MechanicalData(dataset_path, val_annot, classes, train_transforms())
print(train_ds[0])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

pointnet = PointNetCls()

optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)


def train(model, train_loader, val_loader=None, epochs=15, save=True):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = data["pointcloud"].to(device).float()
            labels = data["category"].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(
                    "[Epoch: %d, Batch: %4d / %4d], loss: %.3f"
                    % (epoch + 1, i + 1, len(train_loader), running_loss / 10)
                )
                running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data["pointcloud"].to(device).float(), data[
                        "category"
                    ].to(device)
                    outputs, __, __ = model(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100.0 * correct / total
            print("Valid accuracy: %d %%" % val_acc)

        # save the model
        if save:
            torch.save(model.state_dict(), "save_" + str(epoch) + ".pth")


train(pointnet, train_loader, val_loader, save=False)
