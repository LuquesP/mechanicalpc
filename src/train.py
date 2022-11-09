import torch
from utils.utils import get_classes
from networks.pointnet.pointnet import PointNetCls, feature_transform_regularizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import get_dataset

# from networks.pointnet2.pointnet2_cls_ssg import get_model, get_loss
from networks.pct.init_model import pct_get_model

from networks.repsurf.models.repsurf.repsurf_ssg_umb import Model as repsurfmodel
from networks.repsurf.util.utils import repsurf_get_model, repsurf_get_loss
from networks.repsurf.modules.pointnet2_utils import sample
from networks.repsurf.util.repsurf_args import get_repsurf_args

dataset_path = r"C:\Users\LukasPilsl\source\studium\bachelor\mechanicalpc\data"
model_outpath = "mode/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
device = "cpu"

train_annot, train_ds = get_dataset(dataset_path, "train", False)
val_annot, val_ds = get_dataset(dataset_path, "test", False)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

classes = get_classes(train_annot)
print(len(classes))
# Pointnet
model_naem = "pointnet"
# model = PointNetCls(k=len(classes), feature_transform=True)
model_name = "pointnet2"
# Pointnet2
# model = get_model(len(classes), normal_channel=False)
# PCT
model_name = "pct"
# model = pct_get_model()
# surfrep
repsurfargs = get_repsurf_args()
model = repsurfmodel(repsurfargs)
criterion = repsurf_get_loss()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_batch = len(train_loader) / 64
feature_transform = True
epochs = 1
save = True

for epoch in range(epochs):
    total_train_loss = 0
    correct_examples = 0
    for i, data in enumerate(train_loader, 0):
        points, target = data["pointcloud"], data["category"]
        points = points.transpose(1, 2)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        model = model.train()
        if model_name == "pct":
            pred = model(points)
        elif model_name == "pointnet":
            pred, trans, _ = model(points)
        else:
            pred, trans = model(points)
        # loss = F.nll_loss(pred, target)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.max(1)[1]
        total_train_loss += loss.item()
        correct_examples += pred_choice.eq(target.data).sum().item()
        if i % 10 == 9:
            print(
                f"[Epoch {epoch + 1}, Batch: {i + 1 } / {len(train_loader)}], Train loss: {((total_train_loss / len(train_loader))*100.0):.4f}, train accuracy: {(correct_examples) / len(train_ds):.4f}"
            )
            total_train_loss = 0
    # wandb.log({**metrics})
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            points, target = data["pointcloud"], data["category"]
            points = points.transpose(1, 2)
            points, target = points.to(device), target.to(device)
            outputs, _, _ = model(points)
            _, pred = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        val_acc = 100.0 * correct / total
        print(f"Valid accuracy: {val_acc}")
    if save:
        torch.save(model.state_dict(), "%s/cls_model_%d.pth" % (model_outpath, epoch))
# wandb.finish()
