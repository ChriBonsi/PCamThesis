import random
import time

import h5py
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms

print("starting...")

# Total size of the sets [train, test]: [262144, 32768]
test_set_size = 262144
batch_size = 6500
start_time = time.time()

# Paths to the dataset files
train_images_path = 'data/PCam/train/pcam/camelyonpatch_level_2_split_train_x.h5'
train_labels_path = 'data/PCam/train/pcam/camelyonpatch_level_2_split_train_y.h5'

# Load the HDF5 files
with h5py.File(train_images_path, 'r') as f:
    images = f['x'][:]  # Load all images

with h5py.File(train_labels_path, 'r') as f:
    labels = f['y'][:]  # Load all labels

# Normalize and transform the image for the model
transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Randomly select batch_size number of indices
indices = random.sample(range(len(labels)), batch_size)

# Prepare a batch of images and labels
batch_images = [images[i] for i in indices]
batch_labels = [labels[i] for i in indices]

# Convert images to tensors and stack them into a batch
batch_images_tensor = torch.stack([transform(Image.fromarray(img)) for img in batch_images])

# Load a pre-trained VGG model and modify it for binary classification
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Modify the last layer of the classifier to output 2 classes instead of 1000
num_ftrs = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(num_ftrs, 2)
vgg.eval()

# Perform inference on the batch
with torch.no_grad():
    outputs = vgg(batch_images_tensor)

# Calculate probabilities and predicted classes
probabilities = torch.nn.functional.softmax(outputs, dim=1) * 100
predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()

correct_guesses = 0
wrong_guesses = 0

for i in range(batch_size):
    if (predicted_classes[i] == 1 and batch_labels[i] == 1) or (predicted_classes[i] == 0 and batch_labels[i] == 0):
        correct_guesses += 1
    else:
        wrong_guesses += 1

print(f"\nTest executed on a batch of size {batch_size}, {batch_size / test_set_size * 100:.2f}% of the test set")
print("Time elapsed: {:.2f}s".format(time.time() - start_time))
print(f"Correct guesses: {correct_guesses}")
print(f"Wrong guesses: {wrong_guesses}")
print(f"Accuracy: {correct_guesses / batch_size * 100:.2f}%")
