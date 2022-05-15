"""
Evaluate the trained model on the test data set
with metrics of mAP and accuracy
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Yolov1
from loss import YoloLoss
from dataset import SVHNDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    get_SVHN_accuracy,
)

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
LOAD_MODEL_FILE = "weights/epoch20__mAP0.8333.pth"
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = True
TEST_IMG_DIR = "data_SVHN/dataset/test/"
TEST_LABEL_DIR = "data_SVHN/dataset/test_labels_ratio/"
IOU_THRESHOLD = 0.5
THRESHOLD = 0.4
BOX_FORMAT = "midpoint"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes



if __name__ == "__main__":


    model = Yolov1(split_size=7, num_boxes=2, num_classes=10).to(DEVICE)
    model.eval()

    state_dict = torch.load(LOAD_MODEL_FILE)
    model.load_state_dict(state_dict["state_dict"])
    loss_fn = YoloLoss()
    transform = Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    test_dataset = SVHNDataset(
        "data_SVHN/test.csv", transform=transform, img_dir=TEST_IMG_DIR, label_dir=TEST_LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )


    test_mean_loss = []
    all_pred_boxes = []
    all_true_boxes = []
    img_idx = 0

    print("Evaluating on the test set ...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            batch_size = x.shape[0]
            true_bboxes = cellboxes_to_boxes(y)
            bboxes = cellboxes_to_boxes(out)

            for idx in range(batch_size):
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

            loss = loss_fn(out, y)
            test_mean_loss.append(loss.item())

    accuracy = get_SVHN_accuracy(all_pred_boxes, all_true_boxes)
    print("Accuracy: {}".format(accuracy))

    test_loss = sum(test_mean_loss) / len(test_mean_loss)
    print(f"Test Mean loss was {test_loss}")


    test_pred_boxes, test_target_boxes = get_bboxes(test_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD)
    test_mean_avg_prec = mean_average_precision(test_pred_boxes, test_target_boxes, iou_threshold=IOU_THRESHOLD, box_format="midpoint")

    print(f"Test mAP: {test_mean_avg_prec}")

