"""
Jungle Market - Product/Material Classifier Training Script
Fine-tunes a pretrained MobileNetV3-Small on artisan product images.

Expected folder structure (standard PyTorch ImageFolder format):
    data/images/
        basket/
            img1.jpg
            img2.jpg
        tray/
            img1.jpg
        necklace/
            ...

Usage:
    python train_classifier.py --data_dir ../../data/images --epochs 10
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets


def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Freeze the pretrained feature extractor -- only train the new classifier head
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def train(data_dir, epochs=10, batch_size=16, lr=0.001, output_path="model_weights.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {len(full_dataset)} images across {num_classes} classes: {class_names}")

    if len(full_dataset) < 10:
        print("WARNING: very few images found. Training will run but accuracy will be unreliable "
              "until more labeled images are added to data/images/<category>/.")

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.2%}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, output_path)
    print(f"Model saved to {output_path}")
    return model, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data/images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="model_weights.pt")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data_dir '{args.data_dir}' not found. "
              f"Create it with one subfolder per category, containing labeled images.")
    else:
        train(args.data_dir, args.epochs, args.batch_size, args.lr, args.output)
