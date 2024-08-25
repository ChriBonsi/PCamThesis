import ssl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

ssl._create_default_https_context = ssl._create_unverified_context

# Defining plotting settings
plt.rcParams['figure.figsize'] = 14, 6

# Initializing normalizing transform for the dataset
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Sezione per un'eventuale data augmentation
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
    split='train', download=True
)

full_test_dataset = torchvision.datasets.PCAM(
    root="data/PCam/test", transform=normalize_transform,
    split='test', download=True
)

# Total size of the sets [train, test]: [262144, 32768]
# Define the subset sizes
train_subset_size = 5000
test_subset_size = 1000

# Create indices and subsets for train and test datasets
train_indices, _ = train_test_split(range(len(full_train_dataset)), train_size=train_subset_size, random_state=42)
test_indices, _ = train_test_split(range(len(full_test_dataset)), train_size=test_subset_size, random_state=42)

train_dataset = Subset(full_train_dataset, train_indices)
test_dataset = Subset(full_test_dataset, test_indices)

# Generating data loaders from the corresponding datasets
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Plotting 25 images from the 1st batch
dataiter = iter(train_loader)
images, labels = next(dataiter)

plt.imshow(np.transpose(torchvision.utils.make_grid(images[:25], padding=1, nrow=5).numpy(), (1, 2, 0)))
plt.axis('off')
plt.show()

# Keep the plot window open (not strictly necessary with plt.ion() but may help)
plt.ioff()


# CNN architecture
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 6 * 6, 512),  # Adjusted based on the calculated size
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),  # Binary classification output
            torch.nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.model(x)


# Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

# Defining the model hyperparameters
num_epochs = 15
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training process begins
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0

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
    print(f"Training loss = {train_loss_list[-1]}")

# Plotting loss for all epochs
plt.plot(range(1, num_epochs + 1), train_loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")
plt.show()

# Evaluating the model
test_acc = 0
model.eval()

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)

        outputs = model(images)
        predicted = (outputs > 0.5).float()

        test_acc += (predicted == labels).sum().item()

    print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")

# Plotting a few test images with their predictions
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
predicted = (outputs > 0.5).float()


# Plot images and labels
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Display images with predictions
imshow(torchvision.utils.make_grid(images[:16].cpu()))
print('Actual: ', ' '.join('%5s' % labels[j].item() for j in range(16)))
print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(16)))
