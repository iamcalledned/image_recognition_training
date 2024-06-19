import torch
import torch.optim as optim
import torch.nn as nn
from model import Net

def train_model(trainloader, epochs=25):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        scheduler.step()

    print('Finished Training')
    return net

def save_model(net, path='./models/cnn_model.pth'):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
