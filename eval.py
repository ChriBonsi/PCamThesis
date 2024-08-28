import random

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from vgg19_model import vgg19_binary

# 1. Load images to test the model

# Total size of the sets [train, test]: [262144, 32768]
batch_size = 100

# Paths to the dataset files
train_images_path = 'data/PCam/train/pcam/camelyonpatch_level_2_split_train_x.h5'
train_labels_path = 'data/PCam/train/pcam/camelyonpatch_level_2_split_train_y.h5'

# Load the HDF5 files
with h5py.File(train_images_path, 'r') as f:
    images = f['x'][:]  # Load all images

with h5py.File(train_labels_path, 'r') as f:
    labels = f['y'][:]  # Load all labels

# Normalize and transform the image for the model
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Randomly select batch_size number of indices
indices = random.sample(range(len(labels)), batch_size)

# Prepare a batch of images and labels
batch_images = [images[i] for i in indices]
batch_labels = [labels[i] for i in indices]

# Convert images to tensors and stack them into a batch
batch_images_tensor = torch.stack([transform(Image.fromarray(img)) for img in batch_images])

# 2. Define the model architecture
model = vgg19_binary()  # Don't load the pre-trained weights

# 3. Load the saved model state_dict
model.load_state_dict(torch.load('models/vgg19_25ep.pth'))

# 4. Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Move the images to the same device as the model
batch_images_tensor = batch_images_tensor.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(batch_images_tensor)
    _, predicted_labels = torch.max(outputs, 1)

# Move the predicted labels and actual labels to CPU for visualization
predicted_labels = predicted_labels.cpu().numpy()
actual_labels = np.array(batch_labels).flatten()

# Plotting the results
fig, axes = plt.subplots(10, 10, figsize=(15, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    image = batch_images[i]
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}',
                 color=('green' if actual_labels[i] == predicted_labels[i] else 'red'))

plt.tight_layout()
plt.show()

correct_guesses = 0

for i in range(batch_size):
    if (predicted_labels[i] == 1 and batch_labels[i] == 1) or (predicted_labels[i] == 0 and batch_labels[i] == 0):
        correct_guesses += 1

print(f"Accuracy: {correct_guesses / batch_size * 100:.2f}%")