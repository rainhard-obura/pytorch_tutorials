import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset management and creates mini batches

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 3e-4
batch_size = 32
num_epochs = 10

#Load the data
dataset = CatsAndDogsDataset(
    csv_file='cats_dogs.csv',
    root_dir = 'cats_dogs_resized',
    transform= transforms.ToTensor(),
)

train_set, test_set = torch.utils.data.random_split(dataset, [5,5])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

# Model
model = torchvision.models.googlenet(weights='DEFAULT')

# Freeze all layers and change the final layer with num_classes
for param in model.parameters():
    param.requires_grad = False

# final layer isn't frozen
model.fc = nn.Linear(in_features = 1024, out_features = num_classes)
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

#Train the Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device=device)

        #forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        #append the loss
        losses.append(loss.item())

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/ len(losses)}")

# checking accuracy on training set
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device = device)

            scores = model(x)
            _, num_predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
    model.train()

    print('Checking accuracy on training set')
    check_accuracy(train_loader, model)

    print('checking accuracy on test set')
    check_accuracy(test_loader, model)
