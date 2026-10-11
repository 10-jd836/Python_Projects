from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 25
batch_size = 128

data_dir = Path(__file__).resolve().parent / 'data'
transform = transforms.ToTensor()
training_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
testing_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# -- PLOTTING TRAINING IMAGES
# for i in range(16):
#  plt.subplot(4,4, i+1)
#  plt.xticks([])
#  plt.yticks([])
#  plt.imshow(training_images[i], cmap=plt.cm.binary)
#  plt.xlabel(class_names[training_labels[i][0]])

# plt.show()

# -- BUILDING/TRAINING/COMPILING CNN MODEL
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# -- TRAINING AND EVALUATION
for epoch in range(epochs):
    model.train()
    training_loss = 0.0
    training_correct = 0
    training_count = 0

    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * labels.size(0)
        training_correct += (outputs.argmax(dim=1) == labels).sum().item()
        training_count += labels.size(0)

    model.eval()
    validation_loss = 0.0
    validation_correct = 0
    validation_count = 0
    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * labels.size(0)
            validation_correct += (outputs.argmax(dim=1) == labels).sum().item()
            validation_count += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Loss: {training_loss / training_count:.4f} - "
        f"Accuracy: {training_correct / training_count:.4f} - "
        f"Validation loss: {validation_loss / validation_count:.4f} - "
        f"Validation accuracy: {validation_correct / validation_count:.4f}"
    )

model.eval()
loss = validation_loss / validation_count
accuracy = validation_correct / validation_count
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

torch.save(model.state_dict(), 'image_classifier.pth')

model = ImageClassifier().to(device)
model.load_state_dict(torch.load('image_classifier.pth', map_location=device))
model.eval()

image_files = ['deer.jpg', 'car.jpg', 'horse.jpg', 'plane.jpg']

for filename in image_files:
    img = cv.imread(filename)
    if img is None:
        print(f"Could not read image: {filename}")
        continue

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))

    image_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    index = prediction.argmax(dim=1).item()
    print(f"{filename}: Predicted class is {class_names[index]}")
