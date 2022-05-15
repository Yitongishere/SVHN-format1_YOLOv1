"""
Main file for training Yolo model on SVHN dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import SVHNDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    plot_history,
    cellboxes_to_boxes,
    get_bboxes,
    save_checkpoint,
    get_SVHN_accuracy,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0001
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
SAVE_MODEL_DIR = "weights/"
TRAIN_IMG_DIR = "data_SVHN/dataset/train/"
TRAIN_LABEL_DIR = "data_SVHN/dataset/train_labels_ratio/"
TEST_IMG_DIR = "data_SVHN/dataset/test/"
TEST_LABEL_DIR = "data_SVHN/dataset/test_labels_ratio/"
IOU_THRESHOLD = 0.5
THRESHOLD = 0.3
BOX_FORMAT = "midpoint"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def init_weights(layer):
    # For Conv, normal init
    if type(layer) == torch.nn.Conv2d:
        torch.nn.init.normal_(layer.weight, mean=0, std=0.5)

    # for FC layer, xavier_normal init, bias:0.1
    elif type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0.1)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    all_pred_boxes = []
    all_true_boxes = []
    img_idx = 0

    model.train()
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_bboxes = cellboxes_to_boxes(y)
        bboxes = cellboxes_to_boxes(out)
        for idx in range(BATCH_SIZE):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=IOU_THRESHOLD,
                threshold=THRESHOLD,
                box_format=BOX_FORMAT,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([img_idx] + nms_box)
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > THRESHOLD:
                    all_true_boxes.append([img_idx] + box)

            img_idx += 1

        # update progress bar
        loop.set_postfix(loss=loss.item())

    accuracy = get_SVHN_accuracy(all_pred_boxes, all_true_boxes)
    train_loss = sum(mean_loss)/len(mean_loss)

    return train_loss, accuracy


def val_fn(val_loader, model, loss_fn):
    val_mean_loss = []
    all_pred_boxes = []
    all_true_boxes = []
    img_idx = 0

    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        val_mean_loss.append(loss.item())

        true_bboxes = cellboxes_to_boxes(y)
        bboxes = cellboxes_to_boxes(out)
        for idx in range(BATCH_SIZE):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=IOU_THRESHOLD,
                threshold=THRESHOLD,
                box_format=BOX_FORMAT,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([img_idx] + nms_box)
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > THRESHOLD:
                    all_true_boxes.append([img_idx] + box)

            img_idx += 1

    accuracy = get_SVHN_accuracy(all_pred_boxes, all_true_boxes)
    val_loss = sum(val_mean_loss)/len(val_mean_loss)

    return val_loss, accuracy




if __name__ == "__main__":

    model = Yolov1(split_size=7, num_boxes=2, num_classes=10).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    loss_fn = YoloLoss()

    transform_train = Compose([
                         transforms.ColorJitter(brightness=0.5, saturation=0.5),
                         transforms.RandomGrayscale(p=0.2),
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(), ])
    transform_val = Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = SVHNDataset(
        "data_SVHN/train.csv",
        transform=transform_train,
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
    )

    val_dataset = SVHNDataset(
        "data_SVHN/val.csv",
        transform=transform_val,
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )


    mAP_history = []
    loss_history = []
    accuracy_history = []

    val_mAP_history = []
    val_loss_history = []
    val_accuracy_history = []

    # train the model and validate after each epoch
    for epoch in range(EPOCHS):
        print("------------------------------ Epoch: {} ------------------------------".format(epoch))
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=IOU_THRESHOLD, box_format="midpoint")

        val_pred_boxes, val_target_boxes = get_bboxes(val_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD)
        val_mean_avg_prec = mean_average_precision(val_pred_boxes, val_target_boxes, iou_threshold=IOU_THRESHOLD, box_format="midpoint")

        train_loss, train_accuracy = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Train loss: {train_loss}  ||  Train Accuracy: {train_accuracy}  ||  Train mAP: {mean_avg_prec}")
        with torch.no_grad():
            val_loss, val_accuracy = val_fn(val_loader, model, loss_fn)

            print(f"Val loss: {val_loss}  ||  Val Accuracy: {val_accuracy}  ||  Val mAP: {val_mean_avg_prec}")

        checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
        save_checkpoint(checkpoint, filename=(SAVE_MODEL_DIR + "epoch" + str(epoch+1) + "__mAP" + str(round(mean_avg_prec.item(), 4)) + ".pth"))

        mAP_history.append(mean_avg_prec)
        loss_history.append(train_loss)
        accuracy_history.append(train_accuracy)
        val_mAP_history.append(val_mean_avg_prec)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

    history = {'loss_history':loss_history,
               'val_loss_history':val_loss_history,
               'mAP_history':mAP_history,
               'val_mAP_history':val_mAP_history,
               'accuracy_history':accuracy_history,
               'val_accuracy_history':val_accuracy_history}

    plot_history(history, EPOCHS)
