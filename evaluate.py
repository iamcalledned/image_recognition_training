import torch
from model import Net
from utils import imshow

def load_model(path='./models/cnn_model.pth'):
    net = Net()
    net.load_state_dict(torch.load(path))
    return net

def evaluate_model(net, testloader, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

def predict_images(net, testloader, classes):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(4)))
