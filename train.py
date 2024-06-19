import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from model import Net
from tqdm import tqdm

def train_model(trainloader, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = GradScaler()  # For mixed precision training

    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, data in enumerate(tqdm(trainloader, desc="Training", leave=False)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device
            optimizer.zero_grad()
            
            with autocast():  # Use mixed precision
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        scheduler.step()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Finished Training. Total time: {total_time:.2f} seconds')
    return net

def save_model(net, path='./models/cnn_model.pth'):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
