import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from dataloader import CustomObjectDetectionDataset, get_transform
import zipfile

# ... [Other utility functions like unzip_dataset, train_one_epoch] ...

def evaluate(model, data_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()

    return val_loss / len(data_loader)

def main():
    # Extract zip
    dataset_path = 'path_to_zipped_dataset.zip'
    unzip_dataset(dataset_path, 'dataset_dir')

    # Use our dataset and defined transformations
    dataset = CustomObjectDetectionDataset(root='dataset_dir/train', transforms=get_transform())
    dataset_test = CustomObjectDetectionDataset(root='dataset_dir/val', transforms=get_transform())

    # Split the dataset into train and test sets
    torch.manual_seed(1)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load a pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Track best validation loss and model weights
    best_val_loss = float("inf")

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # Evaluate on the validation set
        current_val_loss = evaluate(model, data_loader_test, device)
        print(f"Epoch {epoch}, Validation Loss: {current_val_loss}")

        # Check if this is the best model so far based on validation loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), "
            torch.save(model.state_dict(), "best_model_weights.pth")
            print(f"Best model weights saved with validation loss: {best_val_loss}")

        # Update the learning rate (if you're using a learning rate scheduler)
        # lr_scheduler.step()

    print("Training completed!")

if __name__ == "__main__":
    main()
