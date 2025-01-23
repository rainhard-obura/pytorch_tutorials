import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setting the hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
load_model = True
# Creating a RNN
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm =nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                        bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagation
        out, _ =self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])
        return out

def save_checkpoint(state,filename ='my_checkpoint.pth.tar'):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
# loading the data
train_dataset = datasets.MNIST(root='dataset/', train= True, transform= transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/',train=False, transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)


# Initializing the network
model =BRNN(input_size,hidden_size, num_layers, num_classes).to(device)

#Loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#if load_model:
 #   load_checkpoint(torch.load('my_checkpoint.pth.tar'))
# Training the Network

for epoch in range(num_epochs):
    #saving the model checkpoints
    if epoch == 2:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data,targets) in enumerate(train_loader):
        # Get data to CUDA if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        # Forward Propagation
        scores = model(data)
        loss = criterion(scores,targets)
        # Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        # Gradient descent or adam step
        optimizer.step()


# check the accuracy on the training set
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training set")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1).view(x.shape[0], sequence_length, input_size)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy \
                {float(num_correct) / float(num_samples) * 100:.2f}%')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)