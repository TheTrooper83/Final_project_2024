import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_accuracy_metric(dataframe, plotname):

    accuracy = dataframe["accuracy"]
    val_accuracy = dataframe["val_accuracy"]
    loss = dataframe["loss"]
    val_loss = dataframe["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # Subplot 1: Training and Validation Accuracy
    axes[0].plot(epochs, accuracy, "bo", label="Training accuracy")
    axes[0].plot(epochs, val_accuracy, "b", label="Validation accuracy")
    axes[0].set_title("Training and validation accuracy " + plotname)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    # plt.figure()
    # Subplot 2: Training and Validation Loss
    axes[1].plot(epochs, loss, "bo", label="Training loss")
    axes[1].plot(epochs, val_loss, "b", label="Validation loss")
    axes[1].set_title("Training and validation loss " + plotname)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()



def generate_confusion_matrix(model_name, dataset):
    
    y_pred = model_name.predict(dataset, verbose=0)

    # https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    plt.figure(figsize=(12,5))

    cm = confusion_matrix(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.show()