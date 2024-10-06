import os
import time
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay, roc_curve, \
    classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from model import CNN

# Initializing normalizing transform for the dataset
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Data augmentation transform
augment_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Downloading the PCam dataset into train and test sets
full_train_dataset = torchvision.datasets.PCAM(
    root="data/PCam/train", transform=normalize_transform,
    split='train', download=False
)

full_test_dataset = torchvision.datasets.PCAM(
    root="data/PCam/test", transform=normalize_transform,
    split='test', download=False
)

# Create an augmented dataset using the augment_transform
augmented_train_dataset = torchvision.datasets.PCAM(
    root="data/PCam/train", transform=augment_transform,
    split='train', download=False
)

start_time = time.time()

# Total size of the sets [train, test]: [262144, 32768]
# Define the subset sizes
train_subset_size = 25000
test_subset_size = 5000

# Create indices and subsets for train and test datasets
train_indices, _ = train_test_split(range(len(full_train_dataset)), train_size=train_subset_size, random_state=42)
test_indices, _ = train_test_split(range(len(full_test_dataset)), train_size=test_subset_size, random_state=42)

# Split the train_dataset into a smaller training set and a validation set
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)  # 80-20 split

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)
test_dataset = Subset(full_test_dataset, test_indices)
augmented_train_dataset = Subset(augmented_train_dataset, train_indices)

# Concatenate the original and augmented datasets
combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmented_train_dataset])

# Generate data loaders for the combined training dataset, validation, and test sets
batch_size = 128
simple_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
combined_train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
used_loader = combined_train_loader # change this

# Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Variables for early stopping
best_loss = float('inf')
best_model_weights = None
patience = 7

# Lists to store the loss values for plotting
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    val_loss = 0

    # Training phase with combined dataset (original + augmented)
    model.train()
    for i, (images, labels) in enumerate(used_loader):
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)  # Reshape labels for binary classification

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss_list.append(train_loss / len(used_loader))

    # Validation phase
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.float().view(-1, 1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss_list.append(val_loss / len(val_loader))

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = deepcopy(model.state_dict())  # Deep copy here
        patience = 7  # Reset patience counter
        print("Patience reset")
    else:
        patience -= 1
        if patience == 0:
            print("Early stopping")
            print("Time elapsed: {:.2f}s".format(time.time() - start_time))
            break

    print(f"Training loss = {train_loss_list[-1]}, Validation loss = {val_loss_list[-1]}")

# Load the best model weights
if best_model_weights:
    model.load_state_dict(best_model_weights)

# Step 3: Calculate and store test metrics (loss, accuracy, confusion matrix, AUC, F1 score)
test_loss = 0
test_acc = 0
all_labels = []
all_outputs = []

model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predicted = (outputs > 0.5).float()
        test_acc += (predicted == labels).sum().item()

        # Store the labels and outputs for later metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")
    print(f"Test set loss = {test_loss}")

# Convert lists to numpy arrays for metric calculations
all_labels = np.array(all_labels).flatten()
all_outputs = np.array(all_outputs).flatten()

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, (all_outputs > 0.5).astype(int))
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.show()

# Calculate the classification report
print("\nClassification Report:")
print(classification_report(all_labels, (all_outputs > 0.5).astype(int)))

# Calculate the AUC and ROC curve for the trained model
auc_trained = roc_auc_score(all_labels, all_outputs)
fpr_trained, tpr_trained, _ = roc_curve(all_labels, all_outputs)
print(f"AUC (Trained): {auc_trained:.4f}")

# Now let's calculate the AUC and ROC curve for the untrained model
untrained_model = CNN().to(device)  # Initialize a new model without loading weights
untrained_outputs = []

untrained_model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = untrained_model(images)
        untrained_outputs.extend(outputs.cpu().numpy())

# Convert the untrained model's outputs to a numpy array
untrained_outputs = np.array(untrained_outputs).flatten()

# Calculate the AUC and ROC curve for the untrained model
auc_untrained = roc_auc_score(all_labels, untrained_outputs)
fpr_untrained, tpr_untrained, _ = roc_curve(all_labels, untrained_outputs)
print(f"AUC (Untrained): {auc_untrained:.4f}")

# Plotting the ROC curve for both trained and untrained models on the same graph
plt.figure()
plt.plot(fpr_trained, tpr_trained, label=f"Trained Model AUC = {auc_trained:.4f}")
plt.plot(fpr_untrained, tpr_untrained, linestyle='--', label=f"Untrained Model AUC = {auc_untrained:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: Trained vs Untrained Model")
plt.legend()
plt.show()

# Calculate the F1 score
f1 = f1_score(all_labels, (all_outputs > 0.5).astype(int))
print(f"F1 Score: {f1:.4f}")

part1 = "abc"
part2 = train_subset_size/1000
if used_loader == combined_train_loader:
    part1 = "aug"
elif used_loader == simple_train_loader:
    part1 = "simple"

# Define a base filename
base_filename = f'new_weights/{part1}_{part2}k.pth'

# Check if the file already exists
if os.path.exists(base_filename):
    # Generate a new filename with a timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = base_filename + "_" + timestamp
else:
    # If file doesn't exist, use the base filename
    filename = base_filename

# Save the model with the unique filename
torch.save(best_model_weights, filename)

# Step 4: Plot the training, validation, and test losses
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label="Training Loss")
plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label="Validation Loss")
plt.axhline(y=test_loss, color='r', linestyle='-', label="Test Loss")  # Test loss as a horizontal line
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("Time elapsed: {:.2f}s".format(time.time() - start_time))
