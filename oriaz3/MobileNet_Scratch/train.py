import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
from torchvision import models

"""
Author: oriaz3

Code adopted from teammate (hyoussfi3@gatech). Following changes made by oriaz3...
- removed pretrain code
"""

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Downloading dataset...")
path = kagglehub.dataset_download("ismailpromus/skin-diseases-image-dataset")

original_dataset_path = os.path.join(path, "IMG_CLASSES")

current_dir = os.path.dirname(os.path.abspath(__file__))
local_dataset_path = os.path.join(current_dir, "skin_dataset")

if not os.path.exists(local_dataset_path):
    os.makedirs(local_dataset_path, exist_ok=True)
    print("Copying dataset to local directory...")
    import shutil
    shutil.copytree(original_dataset_path, local_dataset_path, dirs_exist_ok=True)
    print("Dataset copied successfully.")
else:
    print(f"Local dataset directory already exists at: {local_dataset_path}")

dataset_path = local_dataset_path

class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
for folder in class_folders:
    num_images = len([f for f in os.listdir(os.path.join(dataset_path, folder)) if f.endswith('.jpg')])
    print(f"  - {folder}: {num_images} images")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset...")
full_dataset = ImageFolder(root=dataset_path, transform=None)
print(f"Classes: {full_dataset.classes}")
num_classes = len(full_dataset.classes)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

train_dataset = TransformedSubset(train_dataset, train_transform)
val_dataset = TransformedSubset(val_dataset, val_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

# Define the MobileNetV3 model with custom classifier
def create_mobilenetv3_model(num_classes):
    # Load pre-trained MobileNetV3-Small
    model = models.mobilenet_v3_small(pretrained=False)
    
    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),  # Slightly higher dropout
        nn.Linear(1024, 512),  # Additional layer
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    return model

model = create_mobilenetv3_model(num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': 0.00005}, 
    {'params': model.classifier.parameters(), 'lr': 0.0005}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_dataset)
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_skin_disease_efficientnetb0.pth')
            print(f"Saved new best model with accuracy: {best_val_acc:.4f}")
        print()
    
    return model, history

# Train the model
print("Starting training EfficientNetB0...")
trained_model, history = train_model(model, criterion, optimizer, scheduler)

# Save the final model
torch.save(trained_model.state_dict(), 'skin_disease_efficientnetb0_final.pth')
print("Final model saved to skin_disease_efficientnetb0_final.pth")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('efficientnetb0_training_history.png')
plt.show()

# Evaluate the model
def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = running_loss / len(data_loader.dataset)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_names = full_dataset.classes
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('efficientnetb0_confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy, avg_loss

# Load the best model for evaluation
best_model = create_efficientnet_model(num_classes)
best_model.load_state_dict(torch.load('best_skin_disease_efficientnetb0.pth'))
best_model = best_model.to(device)

print("\nEvaluating best EfficientNetB0 model on validation data...")
val_acc, val_loss = evaluate_model(best_model, val_loader, criterion)
print(f"Best EfficientNetB0 Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")