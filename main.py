from utils import load_data
from train import train_model, save_model
from evaluate import load_model, evaluate_model, predict_images, visualize_misclassifications

def main():
    # Load data
    trainloader, testloader, classes = load_data(batch_size=64, data_root='/mnt/storage')  # Increase batch size

    # Train model
    net = train_model(trainloader, epochs=25)

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
