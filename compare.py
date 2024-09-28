import random
import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from model import CNN  # Assuming the CNN model is defined in model.py

# 1. Load images to test the model

# Total size of the sets [train, test]: [262144, 32768]
batch_size = 5000

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

# Move the images to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_images_tensor = batch_images_tensor.to(device)

# Move the actual labels to CPU for evaluation and plotting
actual_labels = np.array(batch_labels).flatten()

# 2. Define the model architecture
model = CNN()
model = model.to(device)

# List of saved model paths
model_paths = [
    'saved_weights/simple_5k_v2.pth',
    'saved_weights/aug_5k.pth',
    'saved_weights/simple_25k.pth',
    'saved_weights/aug_25k.pth',
    'saved_weights/simple_50k.pth',
    'saved_weights/aug_50k.pth'
]

# Color options for the ROC curves
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Dictionary to store AUC values for each model
model_aucs = {}

# 3. Loop through each model, evaluate and plot ROC curve
plt.figure(figsize=(10, 8))  # Create a single figure for ROC curve

for i, model_path in enumerate(model_paths):
    # Load the saved model state_dict
    model.load_state_dict(torch.load(model_path, mmap=True, weights_only=True))
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(batch_images_tensor)
        predicted_probabilities = outputs.cpu().numpy().flatten()

    # Calculate the AUC for the current model
    auc = roc_auc_score(actual_labels, predicted_probabilities)
    model_aucs[model_path] = auc

    # Compute ROC curve (fpr, tpr)
    fpr, tpr, _ = roc_curve(actual_labels, predicted_probabilities)

    # Plot the ROC curve for the current model on the same plot
    plt.plot(fpr, tpr, label=f'Model {i+1} (AUC = {auc:.4f})', color=colors[i])

# Customize the plot
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess", color='black')  # Diagonal line for reference
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Print out the AUC values for each model
for model_name, auc in model_aucs.items():
    print(f"{model_name}: AUC = {auc:.4f}")
