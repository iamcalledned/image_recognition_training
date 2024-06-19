import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Net
from utils import imshow

def load_trained_model(path='./models/cnn_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, image_path, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = preprocess_image(image_path).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

def main():
    # Load the trained model
    model = load_trained_model()

    # Define the classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Ask the user for the image path
    image_path = input("Please enter the path to the image: ")

    # Make prediction
    predicted_class = predict_image(model, image_path, classes)
    print(f'Predicted class: {predicted_class}')

    # Display the image
    image = Image.open(image_path)
    imshow(transforms.ToTensor()(image))  # Display the image using imshow from utils

if __name__ == "__main__":
    main()
