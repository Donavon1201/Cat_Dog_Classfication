import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random
from collections import Counter

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, num_classes, loss_type='focal', beta=0.9999, gamma=2.0):

        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * num_classes

        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)

        if self.loss_type == 'focal':
            focal_loss = FocalLoss(alpha=weights, gamma=self.gamma)
            return focal_loss(inputs, targets)
        elif self.loss_type == 'softmax':
            return nn.CrossEntropyLoss(weight=weights)(inputs, targets)
        else:
            raise NotImplementedError
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss implementation
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        reduction: 'mean', 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class CIFAR10Classifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(CIFAR10Classifier, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
def create_imbalanced_cifar10(trainset, imbalance_ratio=0.1):
    """
    Args:
        imbalance_ratio: Ratio for minority classes (e.g., 0.1 means 10% of original data)
    """
    targets = np.array(trainset.targets)

    # Classes 0, 2, 3, 5, 7: Keep full dataset (5000 samples each)
    # Classes 1, 4, 6: Keep 50% (2500 samples)
    # Classes 8, 9: Keep 10% (500 samples)

    imbalance_config = {
        0: 1.0,   # airplane - 100%
        1: 0.5,   # automobile - 50%
        2: 1.0,   # bird - 100%
        3: 1.0,   # cat - 100%
        4: 0.5,   # deer - 50%
        5: 1.0,   # dog - 100%
        6: 0.5,   # frog - 50%
        7: 1.0,   # horse - 100%
        8: 0.1,   # ship - 10%
        9: 0.1    # truck - 10%
    }

    indices = []
    for class_idx in range(10):
        class_indices = np.where(targets == class_idx)[0]
        n_samples = int(len(class_indices) * imbalance_config[class_idx])
        selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        indices.extend(selected_indices)

    indices = np.array(indices)
    np.random.shuffle(indices)

    return torch.utils.data.Subset(trainset, indices), imbalance_config


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

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': running_loss/len(dataloader), 'acc': 100*correct/total})

    return running_loss / len(dataloader), 100 * correct / total

# validation
def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return 100 * correct / total, all_preds, all_labels

#calculation of per-class accuracy
def calculate_per_class_accuracy(y_true, y_pred, num_classes=10):
    """Calculate accuracy for each class"""
    per_class_acc = []
    for i in range(num_classes):
        mask = np.array(y_true) == i
        if mask.sum() > 0:
            acc = (np.array(y_pred)[mask] == i).sum() / mask.sum() * 100
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    return per_class_acc

#main train loop
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device, save_path='best_cifar10_model.pth'):
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_acc, _, _ = validate(model, val_loader, device)
        val_accs.append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')

        if scheduler is not None:
            scheduler.step()
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Model saved! Best Val Acc: {best_val_acc:.2f}%')

    return train_losses, train_accs, val_accs

#plotting functions
def plot_class_distribution(targets, title='Class Distribution', save_path='class_distribution.png'):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    counter = Counter(targets)
    classes = [counter[i] for i in range(10)]

    plt.figure(figsize=(12, 6))
    plt.bar(range(10), classes, color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Class distribution plot saved to {save_path}')

def plot_per_class_accuracy(accuracies, title='Per-Class Accuracy', save_path='per_class_accuracy.png'):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(12, 6))
    colors = ['green' if acc > 70 else 'orange' if acc > 50 else 'red' for acc in accuracies]
    plt.bar(range(10), accuracies, color=colors)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Per-class accuracy plot saved to {save_path}')

def infer_preds_labels(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds)

# main execution
if __name__ == '__main__':
    # Configuration
    CONFIG = {
        'batch_size': 128,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'img_size': 224,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_imbalance': True,  # Set to True to create imbalanced dataset
        'imbalance_handling': 'focal_loss'  # Options: 'focal_loss', 'class_balanced'
    }

    print(f"Using device: {CONFIG['device']}")

    #augmentations
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(CONFIG['img_size'], padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #download CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=test_transform)

    #imbalanced dataset creation
    if CONFIG['use_imbalance']:
        print('\nCreating imbalanced dataset...')
        trainset, imbalance_config = create_imbalanced_cifar10(trainset)

        # Get class distribution
        if hasattr(trainset, 'dataset'):
            targets = [trainset.dataset.targets[i] for i in trainset.indices]
        else:
            targets = trainset.targets

        print('\nClass distribution:')
        counter = Counter(targets)
        for i in range(10):
            print(f'Class {i}: {counter[i]} samples')

        plot_class_distribution(targets, 'Imbalanced CIFAR-10 Training Set Distribution')

    train_loader = DataLoader(trainset, batch_size=CONFIG['batch_size'], 
                                 shuffle=True, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(testset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'])

    print(f'\nTrain dataset size: {len(trainset)}')
    print(f'Test dataset size: {len(testset)}')

    # Create model
    model = CIFAR10Classifier(pretrained=True, num_classes=10)
    model = model.to(CONFIG['device'])

    # Select loss function based on imbalance handling method
    if CONFIG['imbalance_handling'] == 'focal_loss':
        criterion = FocalLoss(gamma=2)
        print('\nUsing Focal Loss')
    elif CONFIG['imbalance_handling'] == 'class_balanced':
        if hasattr(trainset, 'dataset'):
            targets = [trainset.dataset.targets[i] for i in trainset.indices]
        else:
            targets = trainset.targets
        counter = Counter(targets)
        samples_per_class = [counter[i] for i in range(10)]
        criterion = ClassBalancedLoss(samples_per_class, num_classes=10, loss_type='focal')
        print('\nUsing Class-Balanced Loss')
    else:
        criterion = nn.CrossEntropyLoss()
        print('\nUsing standard Cross-Entropy Loss')

    # optimizer and LR scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = None #optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])

    # train model
    print('\nStarting training...')
    train_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        scheduler, CONFIG['num_epochs'], CONFIG['device']
    )

    # evaluate on test set
    model.load_state_dict(torch.load('best_cifar10_model.pth'))
    test_acc, test_preds, test_labels = validate(model, test_loader, CONFIG['device'])

    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')

    # calculate per-class accuracy
    per_class_acc = calculate_per_class_accuracy(test_labels, test_preds)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print('\nPer-Class Accuracy:')
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f'{name}: {acc:.2f}%')

    plot_per_class_accuracy(per_class_acc, 'CIFAR-10 Per-Class Test Accuracy')

    # classification report
    print('\nClassification Report:')
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    #plot confusion matrix
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

    cm = confusion_matrix(test_labels, test_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('CIFAR-10 Confusion Matrix')
    plt.tight_layout()
    plt.savefig('cifar10_confusion_matrix.png')  # Saves plot
    plt.show()

    # Save results
    results_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': per_class_acc
    })
    results_df.to_csv('cifar10_results.csv', index=False)
    print('\nResults saved to cifar10_results.csv')