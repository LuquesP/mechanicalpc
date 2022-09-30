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
    running_loss = 0.0
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
        # pred_choice = pred.data.max(1)[1]
        # correct = pred_choice.eq(target.data).cpu().sum()
        running_loss += loss.item()
        if i % 10 == 9:
            metrics = {"train/train_loss": running_loss, "train/epoch": epoch}
            print(
                "[Epoch: %d, Batch: %4d / %4d], loss: %.3f"
                % (epoch + 1, i + 1, len(train_loader), running_loss / 10)
            )
            running_loss = 0.0
            # wandb.log({**metrics})
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in val_loader:
            points, target = data["pointcloud"], data["category"]
            points = points.transpose(1, 2)
            points, target = points.to(device), target.to(device)
            # model = model.eval()
            outputs, _, _ = model(points)
            _, pred = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        val_acc = 100.0 * correct / total
        # correct = pred_choice.eq(target.data).cpu().sum()
        # val_metrics = {
        #     "val/val_loss": loss.item(),
        #     "val/val_accuracy": (correct.item() / float(64)),
        # }

        print("Valid accuracy: %d %%" % val_acc)

    if save:
        torch.save(model.state_dict(), "%s/cls_model_%d.pth" % (model_outpath, epoch))
# wandb.finish()
