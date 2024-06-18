from utils import load_data
from train import train_model, save_model
from evaluate import load_model, evaluate_model, predict_images, visualize_misclassifications

def main():
    # Load data
    trainloader, testloader, classes = load_data(data_root='/mnt/storage')

    # Train model
    net = train_model(trainloader)

    # Save model
    save_model(net)

    # Load model
    net = load_model()

    # Evaluate model
    evaluate_model(net, testloader, classes)

    # Predict new images
    predict_images(net, testloader, classes)

    # Visualize misclassifications
    visualize_misclassifications(net, testloader, classes)

if __name__ == "__main__":
    main()
