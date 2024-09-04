import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from model import CNN

# Initializing normalizing transform for the dataset
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Eventual data augmentation
# augment_transform = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomRotation(10),
#     torchvision.transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# Downloading the PCam dataset into train and test sets
full_train_dataset = torchvision.datasets.PCAM(
    root="data/PCam/train", transform=normalize_transform,
    split='train', download=False
)

full_test_dataset = torchvision.datasets.PCAM(
    root="data/PCam/test", transform=normalize_transform,
    split='test', download=False
)

start_time = time.time()

# Total size of the sets [train, test]: [262144, 32768]
# Define the subset sizes
train_subset_size = 5000
test_subset_size = 1000

# Create indices and subsets for train and test datasets
train_indices, _ = train_test_split(range(len(full_train_dataset)), train_size=train_subset_size, random_state=42)
test_indices, _ = train_test_split(range(len(full_test_dataset)), train_size=test_subset_size, random_state=42)

# Step 1: Split the train_dataset into a smaller training set and a validation set
train_indices, val_indices = train_test_split(train_indices, test_size=0.2,
                                              random_state=42)  # 80-20 split for train-validation

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)
test_dataset = Subset(full_test_dataset, test_indices)

# Generate data loaders for the training, validation, and test sets
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # Training phase
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)  # Reshape labels for binary classification

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss_list.append(train_loss / len(train_loader))

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

# Step 3: Calculate and store test loss
test_loss = 0
test_acc = 0

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

    test_loss /= len(test_loader)
    print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")
    print(f"Test set loss = {test_loss}")

# Save the best model weights
torch.save(best_model_weights, 'saved_weights/simple_5k_v2.pth')

# Step 4: Plot the training, validation, and test losses
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label="Training Loss")
plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label="Validation Loss")
plt.axhline(y=test_loss, color='r', linestyle='-', label="Test Loss")  # Test loss as a horizontal line
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting a few test images with their predictions
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
predicted = (outputs > 0.5).float()

print('Actual: ', ' '.join('%5s' % labels[j].item() for j in range(16)))
print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(16)))
print("Time elapsed: {:.2f}s".format(time.time() - start_time))
