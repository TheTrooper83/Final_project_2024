import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


# Adapted From https://www.tensorflow.org/tutorials/load_data/images
def plot_sample_images(data, labels):

    # Create a new figure with a specific size of 10x10
    plt.figure(figsize=(10, 10))

    # Loop over the first 9 images in the dataset
    for i in range(9):
        
        # Create a subplot in a 3x3 grid
        ax = plt.subplot(3, 3, i + 1)
        
        # Display the i-th image from the 'train_data'    
        plt.imshow(data[:, :, :, i])
        
        # Set the title of the subplot to the corresponding label in 'train_labels'
        plt.title(labels[i])
        
        # Turn off the axis to remove ticks and labels for a cleaner display
        plt.axis("off")

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

def model_comparison_confusion_matrix(model_1_name, model_2_name, dataset, *args):
    
    model_1_y_pred = model_1_name.predict(dataset, verbose=0)
    model_2_y_pred = model_2_name.predict(dataset, verbose=0)

    # https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    model_1_y_true = np.concatenate([y for x, y in dataset], axis=0)
    model_2_y_true = np.concatenate([y for x, y in dataset], axis=0)

    # plt.figure(figsize=(12,5))

    model_1_cm = confusion_matrix(np.argmax(model_1_y_true, axis=1),np.argmax(model_1_y_pred, axis=1))
    model_2_cm = confusion_matrix(np.argmax(model_2_y_true, axis=1),np.argmax(model_2_y_pred, axis=1))

    model_diff = model_1_cm - model_2_cm
    # sns.heatmap(model_diff, annot=True, fmt='d', cmap='viridis', ax=args[0])
    # sns.heatmap(model_diff, annot=True, fmt='d', cmap='viridis')
    sns.heatmap(model_diff, annot=True, fmt='d', cmap='coolwarm')

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    # args[0].set_xlabel ('Predicted Label', fontsize=12)
    # args[0].set_ylabel ('True Label', fontsize=12)
    plt.title('Comparison of the 2 models Confusion Matrices')
    # args[0].set_title('Comparison of the 2 models Confusion Matrices')

    plt.show()


def display_accuracy_loss(history_data, model_name_str, reload=False):

    if  not reload:
        history_pd = pd.DataFrame(history_data.history)
    else:
        history_pd = pd.DataFrame(history_data)

    plot_accuracy_metric(history_pd, model_name_str)

    
def plot_accuracy_metric_comparison(dataframe_1, df1_name_str, dataframe_2, df2_name_str, plotname):

    accuracy_df1 = dataframe_1["accuracy"]
    val_accuracy_df1 = dataframe_1["val_accuracy"]
    loss_df1 = dataframe_1["loss"]
    val_loss_df1 = dataframe_1["val_loss"]
    accuracy_df2 = dataframe_2["accuracy"]
    val_accuracy_df2 = dataframe_2["val_accuracy"]
    loss_df2 = dataframe_2["loss"]
    val_loss_df2 = dataframe_2["val_loss"]
    epochs = range(1, len(accuracy_df1) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # Subplot 1: Training and Validation Accuracy
    axes[0].plot(epochs, accuracy_df1, "bo", label= df1_name_str + ": Training accuracy")
    axes[0].plot(epochs, val_accuracy_df1, "b", label= df1_name_str +": Validation accuracy")
    axes[0].plot(epochs, accuracy_df2, "o", color='orange', label= df2_name_str +": Training accuracy" )
    axes[0].plot(epochs, val_accuracy_df2, "orange", label= df2_name_str + ": Validation accuracy")
    
    axes[0].set_title("Training and validation accuracy " + plotname)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    # plt.figure()
    # Subplot 2: Training and Validation Loss
    axes[1].plot(epochs, loss_df1, "bo", label=df1_name_str + ": Training loss ")
    axes[1].plot(epochs, val_loss_df1, "b", label=df1_name_str + ": Validation loss ")
    axes[1].plot(epochs, loss_df2, "o", color='orange', label=df2_name_str + ": Training loss ")
    axes[1].plot(epochs, val_loss_df2, "orange", label=df2_name_str + ": Validation loss ")
    axes[1].set_title("Training and validation loss " + plotname)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def confusion_matrix_misclassification(model_name, dataset):
    
    y_pred = model_name.predict(dataset, verbose=0)

    # https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    plt.figure(figsize=(12,5))

    cm = confusion_matrix(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))

    # https://stackoverflow.com/questions/56084882/how-to-show-precision-in-a-confusion-matrix
    return np.sum(cm) - np.trace(cm)

def confusion_matrix_avarage_correct_classifications(model_name, dataset):
    
    y_pred = model_name.predict(dataset, verbose=0)

    # https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    plt.figure(figsize=(12,5))

    cm = confusion_matrix(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))

    # https://stackoverflow.com/questions/56084882/how-to-show-precision-in-a-confusion-matrix
    return np.diag(cm)