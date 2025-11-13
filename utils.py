
"""
Utility functions for image analysis and model interpretation
IE4483 Mini Project
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import cv2


def visualize_predictions(model, dataset, device, num_samples=16, save_path='predictions_viz.png'):
    """
    Visualize model predictions on random samples
    """
    model.eval()

    # Randomly select samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    class_names = ['Cat', 'Dog']  # For Dogs vs Cats

    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

            # Denormalize image for display
            img_display = image.cpu().numpy().transpose(1, 2, 0)
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)

            ax.imshow(img_display)
            true_label = class_names[label]
            pred_label = class_names[predicted.item()]

            color = 'green' if label == predicted.item() else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Predictions visualization saved to {save_path}')


def visualize_cifar10_predictions(model, dataset, device, num_samples=25, save_path='cifar10_predictions.png'):
    """
    Visualize CIFAR-10 model predictions
    """
    model.eval()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()

    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            # Denormalize
            img_display = image.cpu().numpy().transpose(1, 2, 0)
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)

            ax.imshow(img_display)
            true_label = class_names[label]
            pred_label = class_names[predicted.item()]
            confidence = probabilities[0][predicted.item()].item() * 100

            color = 'green' if label == predicted.item() else 'red'
            ax.set_title(f'T: {true_label}\nP: {pred_label} ({confidence:.1f}%)', 
                        color=color, fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'CIFAR-10 predictions visualization saved to {save_path}')


def analyze_misclassifications(model, dataset, device, num_examples=10, save_path='misclassifications.png'):
    """
    Analyze and visualize misclassified examples
    """
    model.eval()

    misclassified = []
    class_names = ['Cat', 'Dog']

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

            if predicted.item() != label:
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities[0][predicted.item()].item()
                misclassified.append((idx, label, predicted.item(), confidence, image))

                if len(misclassified) >= num_examples:
                    break

    if len(misclassified) == 0:
        print("No misclassifications found!")
        return

    # Visualize
    n_cols = 5
    n_rows = (len(misclassified) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, true_label, pred_label, conf, image) in enumerate(misclassified):
        row = idx // n_cols
        col = idx % n_cols

        # Denormalize
        img_display = image.cpu().numpy().transpose(1, 2, 0)
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)

        axes[row, col].imshow(img_display)
        axes[row, col].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({conf*100:.1f}%)', 
                                 fontsize=10)
        axes[row, col].axis('off')

    # Hide empty subplots
    for idx in range(len(misclassified), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Misclassifications analysis saved to {save_path}')


def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """
    Plot ROC curve for binary classification
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'ROC curve saved to {save_path}')


def plot_precision_recall_curve(y_true, y_scores, save_path='pr_curve.png'):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Precision-Recall curve saved to {save_path}')


def visualize_feature_maps(model, image, device, layer_name='layer1', save_path='feature_maps.png'):
    """
    Visualize feature maps from a specific layer
    """
    activations = {}

    def hook_fn(module, input, output):
        activations['features'] = output

    # Register hook
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, layer_name):
            hook = getattr(model.backbone, layer_name).register_forward_hook(hook_fn)
        else:
            print(f"Layer {layer_name} not found")
            return
    else:
        if hasattr(model, layer_name):
            hook = getattr(model, layer_name).register_forward_hook(hook_fn)
        else:
            print(f"Layer {layer_name} not found")
            return

    # Forward pass
    model.eval()
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        _ = model(image_tensor)

    hook.remove()

    # Visualize feature maps
    features = activations['features'].cpu().numpy()[0]
    n_features = min(16, features.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for idx in range(n_features):
        axes[idx].imshow(features[idx], cmap='viridis')
        axes[idx].set_title(f'Feature {idx}')
        axes[idx].axis('off')

    for idx in range(n_features, 16):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Feature maps saved to {save_path}')


def analyze_training_dynamics(train_losses, train_accs, val_losses, val_accs, 
                              save_path='training_dynamics.png'):
    """
    Detailed analysis of training dynamics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy curves
    axes[0, 1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Overfitting analysis (train - val)
    loss_gap = [t - v for t, v in zip(train_losses, val_losses)]
    axes[1, 0].plot(loss_gap, linewidth=2, color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Gap (Train - Val)')
    axes[1, 0].set_title('Overfitting Analysis (Loss)')
    axes[1, 0].grid(True)

    # Generalization gap
    acc_gap = [t - v for t, v in zip(train_accs, val_accs)]
    axes[1, 1].plot(acc_gap, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (Train - Val) %')
    axes[1, 1].set_title('Generalization Gap Analysis')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training dynamics analysis saved to {save_path}')


def create_submission_file(image_names, predictions, filename='submission.csv'):
    """
    Create submission file in the required format
    """
    import pandas as pd

    df = pd.DataFrame({
        'ID': image_names,
        'Label': predictions
    })
    df.to_csv(filename, index=False)
    print(f'Submission file created: {filename}')
    print(f'Total predictions: {len(predictions)}')
    print(f'Dogs (1): {sum(predictions)}, Cats (0): {len(predictions) - sum(predictions)}')


if __name__ == '__main__':
    print("Utility functions for IE4483 Mini Project")
    print("Import these functions in your main training scripts for visualization and analysis")
