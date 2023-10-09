import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from dataloader import CustomObjectDetectionDataset, get_transform
from sklearn.metrics import precision_recall_curve, average_precision_score

CLASSES = ['background', 'duckie', 'bot', 'traffic_sign', 'other']

def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_predictions(model, data_loader, device):
    all_predictions = []
    all_ground_truths = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)

            all_predictions.extend(predictions)
            all_ground_truths.extend(targets)

    return all_predictions, all_ground_truths

def compute_map(predictions, ground_truths):
    # This function now computes precision and recall for each class
    scores_per_class = {cls: [] for cls in CLASSES[1:]}
    ground_truths_binary_per_class = {cls: [] for cls in CLASSES[1:]}
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes'].tolist()
        pred_scores = pred['scores'].tolist()
        pred_labels = pred['labels'].tolist()
        
        gt_boxes = gt['boxes'].tolist()
        gt_labels = gt['labels'].tolist()

        for cls in CLASSES[1:]:
            class_id = CLASSES.index(cls)
            for pbox, pscore, plabel in zip(pred_boxes, pred_scores, pred_labels):
                if plabel == class_id:
                    match = False
                    for gbox, glabel in zip(gt_boxes, gt_labels):
                        if glabel == class_id:
                            iou = # compute Intersection Over Union between pbox and gbox
                            if iou > 0.5:
                                match = True
                                break
                    scores_per_class[cls].append(pscore)
                    ground_truths_binary_per_class[cls].append(1 if match else 0)

    precision_recall_output = {}
    for cls in CLASSES[1:]:
        precision, recall, _ = precision_recall_curve(ground_truths_binary_per_class[cls], scores_per_class[cls])
        average_precision = average_precision_score(ground_truths_binary_per_class[cls], scores_per_class[cls])
        precision_recall_output[cls] = (precision, recall, average_precision)

    return precision_recall_output

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = load_model('best_model_weights.pth', device)

    # Prepare data
    dataset_test = CustomObjectDetectionDataset(root='dataset_dir/val', transforms=get_transform())
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # Get predictions
    predictions, ground_truths = get_predictions(model, data_loader_test, device)

    # Compute mAP
    precision_recall_output = compute_map(predictions, ground_truths)
    
    # Print AP for each class and plot Precision-Recall curve
    for cls, (precision, recall, average_precision) in precision_recall_output.items():
        print(f"AP for {cls}: {average_precision}")
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve for {cls}: AP={average_precision:0.2f}')
        plt.show()

if __name__ == "__main__":
    main()
