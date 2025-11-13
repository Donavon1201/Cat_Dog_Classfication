import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import random
from utils import (
    visualize_predictions,
    analyze_misclassifications,
    plot_roc_curve,
    plot_precision_recall_curve,
    analyze_training_dynamics,
)

#define random seed so that we can reproduce results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#custom dataset class
class DogCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        #extract dog and cat images
        for class_name in ['dog', 'cat']:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(1 if class_name == 'dog' else 0)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
#custom class for test dataset with no labels
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        #extract all the test images
        for img_name in sorted(os.listdir(root_dir)):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(root_dir, img_name))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)


#using resnet18 model with transfer learning
class DogCatClassifier(nn.Module):
    def __init__(self, pretrained = True, num_classes=2):
        super(DogCatClassifier, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
#training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss':running_loss/total, 'acc': 100*correct/total})

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

#validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels

#main training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_path='best_model.pth', patience=5):
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, mode='max', verbose=True)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-'*60)

        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train ACC: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.2f}%')

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Model with best accuracy saved. Best Val Acc: {best_val_acc:.2f}%')

        # Check early stopping condition
        if early_stopping(val_acc):  # You can also use val_loss instead of val_acc here
            print("Early stopping triggered.")
            break  # Stop training if early stopping is triggered

    return train_losses, train_accs, val_losses, val_accs

#predict function for testing set
def predict_test(model, test_loader, device):
    model.eval()
    predictions = []
    image_names = []

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            image_names.extend(names)
    
    return image_names, predictions

#plotting function
def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

    #loss plot
    ax1.plot(train_losses, label='Train loss')
    ax1.plot(val_losses, label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    #accuracy plot
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training history plot saved to {save_path}')

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')


#main python fucntion

if __name__ == '__main__':
    # Configuration
    CONFIG = {
        'data_dir': './dataset', 
        'train_dir': './dataset/train',
        'val_dir': './dataset/val',
        'test_dir': './dataset/test',
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'img_size': 224,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Using device: {CONFIG['device']}")
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),

        # Rotation without specifying interpolation mode
        transforms.RandomRotation(
            degrees=25,
            fill=(128, 128, 128)   # you can remove this if your torchvision is too old
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # Blurring
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=0.35
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #create the dataset
    train_dataset = DogCatDataset(CONFIG['train_dir'], transform=train_transform)
    val_dataset = DogCatDataset(CONFIG['val_dir'], transform=val_transform)
    test_dataset = TestDataset(CONFIG['test_dir'], transform=val_transform)


    #print number of images being use to train validate and test
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    #create dataloader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'])
    
    #create model
    model = DogCatClassifier(pretrained=True, num_classes=2)
    model = model.to(CONFIG['device'])

    #Early Stopping
    class EarlyStopping:
        def __init__(self, patience=5, mode='min', verbose=True, delta=0):
            """
            Args:
                patience (int): How many epochs to wait for improvement before stopping.
                mode (str): 'min' for minimizing loss, 'max' for maximizing accuracy.
                verbose (bool): Whether to print messages about early stopping.
                delta (float): Minimum change to qualify as an improvement.
            """
            self.patience = patience
            self.mode = mode
            self.verbose = verbose
            self.delta = delta
            
            # Initialize counters and best performance
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def __call__(self, metric):
            if self.best_score is None:
                self.best_score = metric
            elif (self.mode == 'min' and metric < self.best_score - self.delta) or \
                (self.mode == 'max' and metric > self.best_score + self.delta):
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered. No improvement for {self.patience} epochs.")
            return self.early_stop

    #define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],weight_decay=1e-4)

    #StepLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #train the model
    print('\nStarting training...')
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, CONFIG['num_epochs'], CONFIG['device']
    )
    #training history plot
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    # evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    _, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, CONFIG['device'])

    print(f'\nFinal Validation Accuracy: {val_acc:.2f}%')

    # plot the confusion matrix
    plot_confusion_matrix(val_labels, val_preds)

    # classification report
    print('\nClassification Report:')
    print(classification_report(val_labels, val_preds, target_names=['Cat', 'Dog']))

    # predition on test set
    print('\nPredicting on test set...')
    image_names, predictions = predict_test(model, test_loader, CONFIG['device'])

    # Show first few misclassified examples from validation data
    analyze_misclassifications(model, val_dataset, CONFIG['device'], num_examples=15, save_path='misclassifications.png')


    # submission file in excel
    submission_df = pd.DataFrame({
        'ID': [os.path.splitext(name)[0] for name in image_names],
        'Label': predictions
    })
    submission_df.to_csv('submission.csv', index=False)
    print('Submission file saved to submission.csv')
