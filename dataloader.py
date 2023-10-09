import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # Load image and annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "annotations", self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Here, we'll read the annotation file and extract bounding boxes
        boxes = []
        labels = []
        with open(label_path, "r") as file:
            for line in file:
                xmin, ymin, xmax, ymax, label = line.strip().split(',')
                boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                labels.append(int(label))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # There's only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform():
    return T.Compose([
        T.ToTensor(),
    ])
