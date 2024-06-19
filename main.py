from utils import load_data
from train import train_model, save_model
from evaluate import load_model, evaluate_model, predict_images, visualize_misclassifications

def main():
    # Load data
    print("Loading data...")
    trainloader, testloader, classes = load_data(data_root='/mnt/storage')
    print("Data loaded")

    # Train model
    print("Starting training...")
    net = train_model(trainloader, epochs=25)
    print("Training finished")

    # Save model
    save_model(net)
    print("Model saved")

    # Load model
    net = load_model()
    print("Model loaded")

    # Evaluate model
    evaluate_model(net, testloader, classes)
    print("Evaluation done")

    # Predict new images
    predict_images(net, testloader, classes)
    print("Prediction done")

    # Visualize misclassifications
    visualize_misclassifications(net, testloader, classes)
    print("Misclassifications visualized")

if __name__ == "__main__":
    main()

