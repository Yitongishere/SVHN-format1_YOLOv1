"""
make predictions on a serials of images
Put the images into './predict_examples' directory!
"""
import os
import torch
from PIL import Image
from model import Yolov1
import torchvision.transforms as transforms
from utils import cellboxes_to_boxes, non_max_suppression, plot_image


LOAD_MODEL_FILE = "weights/epoch20__mAP0.8333.pth"
PREDICT_IMG_DIR = "predict_examples/"

if __name__ == "__main__":
    model = Yolov1(split_size=7, num_boxes=2, num_classes=10)
    state_dict = torch.load(LOAD_MODEL_FILE)
    model.load_state_dict(state_dict["state_dict"])
    model.eval()
    with torch.no_grad():
        images = []
        sizes = []
        trans1 = transforms.Resize((224, 224))
        trans2 = transforms.ToTensor()
        for _, _, files in os.walk("predict_examples/"):
            for file in files:
                img_path = PREDICT_IMG_DIR + file
                image = Image.open(img_path)

                ori_width = image.width
                ori_height = image.height
                sizes.append([ori_width, ori_height])

                image = trans1(image)
                image = trans2(image)
                image = image.tolist()
                images.append(image)
        images = torch.tensor(images)



        pred_cellboxes = model(images)
        for idx in range(images.shape[0]):
            bboxes = cellboxes_to_boxes(pred_cellboxes)
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image(images[idx].permute(1, 2, 0), bboxes, sizes[idx][0]*5, sizes[idx][1]*5)




