import torch
from tqdm import tqdm

# model = PointNetCls()


def eval_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(dataloader, 0)):
        points, target = data["pointcloud"], data["category"]
        # target = target[:, 0]
        points = points.transpose(1, 2)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("final accuracy {}".format(total_correct / float(total_testset)))
