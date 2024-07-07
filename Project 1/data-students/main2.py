import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re

# 1. Load and process the data
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset_path = './TRAIN'
test_dataset_path = './TEST'

BATCH_SIZE = 32


train_datagen = datasets.ImageFolder(root=train_dataset_path, transform=transform)
train_dataset_loader = DataLoader(train_datagen, batch_size=BATCH_SIZE, shuffle=True)

# Assuming labels.csv maps numeric labels to class names
the_real_labels = {}
with open("./labels.csv", "r") as label_f:
    for line in label_f.readlines()[1:]:
        label_value, label_description = line.strip().split(";")
        the_real_labels[int(label_value)] = label_description

print("Label Mappings for classes present in the training and validation datasets\n")
#for key, value in train_datagen.class_to_idx.items():
 #   print(f"{key} : {value} - {the_real_labels[value]}")

# 2. Define the Neural Network Architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(75*75*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = x.view(-1, 75*75*3)  # Flatten the input tensor
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*18*18, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = x.view(-1, 32*18*18)  # Flatten the output
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Training Function
def fit(data_loader, model, criterion, optimizer, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(n_epochs):
        accu_loss = 0
        all_preds = []
        all_targets = []
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            accu_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_preds.extend(y_pred.argmax(dim=1).tolist())
            all_targets.extend(y.tolist())

        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {accu_loss:.4f}, Balanced Accuracy: {balanced_acc * 100:.2f}%')

# 4. Initialize Models, Loss Functions, Optimizers
mlp_model = MLP()
cnn_model = CNN()

criterion = nn.CrossEntropyLoss()

optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)

# 5. Train the models
n_epochs = 10

print("Training MLP Model")
fit(train_dataset_loader, mlp_model, criterion, optimizer_mlp, n_epochs)

print("\nTraining CNN Model")
fit(train_dataset_loader, cnn_model, criterion, optimizer_cnn, n_epochs)


class UnlabeledDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Load the test dataset
test_dataset = UnlabeledDataset(images_folder=test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to make predictions
def predict(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for X in data_loader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Make predictions with CNN and MLP models
cnn_predictions = predict(cnn_model, test_loader)
mlp_predictions = predict(mlp_model, test_loader)

# At this point, `cnn_predictions` and `mlp_predictions` will hold the predicted class indices

import matplotlib.pyplot as plt


def visualize_predictions(model, data_loader, class_labels, num_images=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    images_so_far = 0
    plt.figure(figsize=(10, 20))

    with torch.no_grad():
        for i, (inputs) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 5, 5, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_labels[preds[j].item()]}')
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)

                if images_so_far == num_images:
                    plt.show()
                    return


# Assuming the_real_labels maps from class indices to labels
# Adjust the_real_labels if necessary to match model output
class_labels = [the_real_labels[i] for i in sorted(the_real_labels.keys())]

# Visualize predictions for CNN and MLP models
print("CNN Model Predictions on Test Data:")
visualize_predictions(cnn_model, test_loader, class_labels, num_images=10)

print("MLP Model Predictions on Test Data:")
visualize_predictions(mlp_model, test_loader, class_labels, num_images=10)
