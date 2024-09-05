import random

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, RocCurveDisplay, ConfusionMatrixDisplay

from model import CNN

# 1. Load images to test the model

# Total size of the sets [train, test]: [262144, 32768]
batch_size = 100

# Paths to the dataset files
eval_images_path = 'data/PCam/test/pcam/camelyonpatch_level_2_split_test_x.h5'
eval_labels_path = 'data/PCam/test/pcam/camelyonpatch_level_2_split_test_y.h5'

# Load the HDF5 files
with h5py.File(eval_images_path, 'r') as f:
    images = f['x'][:]  # Load all images

with h5py.File(eval_labels_path, 'r') as f:
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
model = CNN()

# 3. Load the saved model state_dict
model.load_state_dict(torch.load('saved_weights/simple_5k_v2.pth'))

# 4. Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Move the images to the same device as the model
batch_images_tensor = batch_images_tensor.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(batch_images_tensor)
    predicted_labels = (outputs > 0.5).float().cpu().numpy()

# Move the actual labels to CPU for visualization
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

# Calculate the confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.show()

# Calculate the AUC
auc = roc_auc_score(actual_labels, outputs.cpu().numpy())
print(f"AUC: {auc:.4f}")
RocCurveDisplay.from_predictions(actual_labels, outputs.cpu().numpy())
plt.title("ROC Curve")
plt.show()

# Calculate the F1 score
f1 = f1_score(actual_labels, predicted_labels)
print(f"F1 Score: {f1:.4f}")

# Calculate and print the accuracy
correct_guesses = np.sum(predicted_labels.flatten() == actual_labels)
print(f"Accuracy: {correct_guesses / batch_size * 100:.2f}%")
