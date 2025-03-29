import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# =======================================================================================================

# Adapted From https://www.tensorflow.org/tutorials/load_data/images
def plot_sample_images(data, labels):
    """
    It shows the first 9 pictures from a dataset along with their labels in a 3x3 grid.

    Args:
        data (numpy.ndarray): A 4D array of shape (height, width, channels, num_images),
                            where images are accessed using data[:, :, :, i].
        labels (list or array-like): A list of labels corresponding to each image.

    Output:
        Display 9 Images in a grid

    """

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

# =======================================================================================================

def plot_accuracy_metric(dataframe, plotname):
    """
    Plots training and validation accuracy and loss over epochs.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing training history
        plotname (str): A string to include in the plot titles for identification

    Output:
        Displays 2 plots showing Training and Validation,
        Accuracy and loss information
    
    """

    # Extract training metrics from the DataFrame
    accuracy = dataframe["accuracy"]
    val_accuracy = dataframe["val_accuracy"]
    loss = dataframe["loss"]
    val_loss = dataframe["val_loss"]

     # X-axis range based on number of epochs
    epochs = range(1, len(accuracy) + 1)

    # Create a 1x2 subplot for accuracy and loss
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
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# =======================================================================================================

def generate_confusion_matrix(model_name, dataset):
    """
    Generates a confusion matrix heatmap for a given model and dataset

    Args:
        model_name (tf.keras.Model): Trained model used to generate predictions.
        dataset (tf.data.Dataset): Dataset containing image-label pairs

    Output:
        Displays a heatmap showing the number of correct and incorrect predictions
        for each class
    """

    # Generate model predictions for the entire dataset
    y_pred = model_name.predict(dataset, verbose=0)

    # Extract the true labels from the dataset
    # For moe information: https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    # Create a new figure for the heatmap
    plt.figure(figsize=(12,5))

     # Compute confusion matrix using predicted and true labels
    cm = confusion_matrix(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))

    # Display the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')

     # Set axis labels
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Show the plot
    plt.show()

# =======================================================================================================

def model_comparison_confusion_matrix(model_1_name, model_2_name, dataset, *args):
    """
    Finds the difference between two classification models
    confusion matrices on a given dataset and displays it.

    Args:
        model_1_name (tf.keras.Model): First trained model to compare.
        model_2_name (tf.keras.Model): Second trained model to compare.
        dataset (tf.data.Dataset): Dataset containing (image, label) pairs.
        *args: Optional matplotlib axis object for plotting externally.

    Output:
        Displays a heatmap showing where and how the models differ in classification
        performance, with red/blue indicating one model outperforming the other.
    
    """
    
     # Generate predictions for both models
    model_1_y_pred = model_1_name.predict(dataset, verbose=0)
    model_2_y_pred = model_2_name.predict(dataset, verbose=0)

    # Extract the true labels from the dataset (same for both models)
    # For more information: https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    model_1_y_true = np.concatenate([y for x, y in dataset], axis=0)
    model_2_y_true = np.concatenate([y for x, y in dataset], axis=0)

    # plt.figure(figsize=(12,5))

    # Compute confusion matrices for each model
    model_1_cm = confusion_matrix(np.argmax(model_1_y_true, axis=1),np.argmax(model_1_y_pred, axis=1))
    model_2_cm = confusion_matrix(np.argmax(model_2_y_true, axis=1),np.argmax(model_2_y_pred, axis=1))

    # Compute the difference between the two confusion matrices
    model_diff = model_1_cm - model_2_cm

    # Plot the difference as a heatmap
    sns.heatmap(model_diff, annot=True, fmt='d', cmap='coolwarm')

    # Set axis labels and title
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Comparison of the 2 models Confusion Matrices')

    # Show the plot
    plt.show()

# =======================================================================================================


def display_accuracy_loss(history_data, model_name_str, reload=False):
    """
    Displays training and validation accuracy/loss plots for a given model.

    Args:
        history_data (tf.keras.callbacks.History or pd.DataFrame): 
            The training history object returned by model.fit(), or a pre-saved DataFrame.

        model_name_str (str): Name of the model to use in the plot title.

        reload (bool): If True, treat `history_data` as a DataFrame; 
                       otherwise, extract from history_data.history.

    Output:
         Visualize accuracy and loss curves.

    """

    # Convert training history to a DataFrame, depending on whether it's reloaded or fresh
    if  not reload:
        history_pd = pd.DataFrame(history_data.history)
    else:
        history_pd = pd.DataFrame(history_data)

    # Generate accuracy/loss plots using the provided model name
    plot_accuracy_metric(history_pd, model_name_str)

# =======================================================================================================

    
def plot_accuracy_metric_comparison(dataframe_1, df1_name_str, dataframe_2, df2_name_str, plotname):
    """

    Compares the accuracy and loss of training and evaluation for two different models over epochs.

    Args:
        dataframe_1 (pd.DataFrame): Training history of the first model.
        df1_name_str (str): Label to use in the legend for the first model.
        dataframe_2 (pd.DataFrame): Training history of the second model.
        df2_name_str (str): Label to use in the legend for the second model.
        plotname (str): String to append in plot titles

    Output:
        Training and Validation Accuracy and loss comparison
    
    """
    # Extract accuracy and loss metrics from both dataframes
    accuracy_df1 = dataframe_1["accuracy"]
    val_accuracy_df1 = dataframe_1["val_accuracy"]
    loss_df1 = dataframe_1["loss"]
    val_loss_df1 = dataframe_1["val_loss"]
    accuracy_df2 = dataframe_2["accuracy"]
    val_accuracy_df2 = dataframe_2["val_accuracy"]
    loss_df2 = dataframe_2["loss"]
    val_loss_df2 = dataframe_2["val_loss"]

     # Generate an epoch range for plotting
    epochs = range(1, len(accuracy_df1) + 1)

     # Create side-by-side subplots
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
    
    # Improve layout and display the plots
    plt.tight_layout()
    plt.show()

# =======================================================================================================


def confusion_matrix_misclassification(model_name, dataset):
    """
    Finds out how many wrong labels a model made on a given dataset.

    Args:
        model_name (tf.keras.Model): Trained model to evaluate.
        dataset (tf.data.Dataset): Dataset containing (image, label) pairs.

    Returns:
        int: Total number of misclassified predictions
    
    """
    
    # Generate predictions for the dataset
    y_pred = model_name.predict(dataset, verbose=0)

    # Extract true labels from the dataset
    # For more information: https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    # Create a figure for plotting
    plt.figure(figsize=(12,5))

    # Compute confusion matrix from true and predicted labels
    cm = confusion_matrix(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))

    # Calculate total misclassifications (total - correct predictions)
    # For more information: https://stackoverflow.com/questions/56084882/how-to-show-precision-in-a-confusion-matrix
    return np.sum(cm) - np.trace(cm)

# =======================================================================================================


def confusion_matrix_avarage_correct_classifications(model_name, dataset):
    """
    Calculates the number of right classifications for each class

    Args:
        model_name (tf.keras.Model): A trained classification model.
        dataset (tf.data.Dataset): Dataset consisting of (image, label) pairs.

    Returns:
        np.ndarray: A 1D array containing the number of correct predictions for each class.

    
    """
    # Generate predictions for the dataset
    y_pred = model_name.predict(dataset, verbose=0)

    # Extract the true labels from the dataset
    # For more information: https://stackoverflow.com/questions/56226621/how-to-extract-data-labels-back-from-tensorflow-dataset 
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    # Create a new figure 
    plt.figure(figsize=(12,5))

    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))

    # Return the diagonal of the confusion matrix
    # For more information: https://stackoverflow.com/questions/56084882/how-to-show-precision-in-a-confusion-matrix
    return np.diag(cm)