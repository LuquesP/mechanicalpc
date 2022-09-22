import torch
from utils.utils import get_classes
from models.pointnet import PointNetCls, feature_transform_regularizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import get_dataset

dataset_path = r"C:\Users\LukasPilsl\source\studium\bachelor\mechanicalpc\data"
model_outpath = "mode/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_annot, train_ds = get_dataset(dataset_path, "train")
val_annot, val_ds = get_dataset(dataset_path, "test")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

classes = get_classes(train_annot)
model = PointNetCls(k=len(classes), feature_transform=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_batch = len(train_loader) / 64
feature_transform = True
epochs = 1
save = True

for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        points, target = data["pointcloud"], data["category"]
        points = points.transpose(1, 2)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        model = model.train()
        pred, trans, trans_feat = model(points)
        loss = F.nll_loss(pred, target)
        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print(
            "[%d: %d/%d] train loss: %f accuracy: %f"
            % (epoch, i, len(train_loader), loss.item(), correct.item() / float(64))
        )
        if i % 10 == 0:
            j, data = next(enumerate(val_loader, 0))
            points, target = data["pointcloud"], data["category"]
            points = points.transpose(1, 2)
            points, target = points.to(device), target.to(device)
            model = model.eval()
            pred, _, _ = model(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print(
                "[%d: %d/%d] %s loss: %f accuracy: %f"
                % (
                    epoch,
                    i,
                    num_batch,
                    ("test"),
                    loss.item(),
                    correct.item() / float(64),
                )
            )

    if save:
        torch.save(model.state_dict(), "%s/cls_model_%d.pth" % (model_outpath, epoch))
