from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import  Model
import pandas as pd
import numpy as np
import tensorflow as tf

# =======================================================================================================

def label_count_plot(labels, x_axis, y_axis, title, axis=None):
    """
    Creates a bar plot to visualize the count of unique labels in a dataset.

    Parameters:
    -----------
    labels : array-like
        A list, array, or similar structure containing categorical data.
    x_axis : str
        The label for the x-axis of the plot.
    y_axis : str
        The label for the y-axis of the plot.
    title : str
        The title of the bar plot.

    Returns: None
    --------
    """

    # Get unique labels and their counts from the input data
    unqique, counts = np.unique(labels, return_counts=True)

    # Combine the unique labels and their counts into a dictionary
    dict_counts =  dict(zip(unqique, counts))

    # Create a bar plot using Seaborn
    # x-axis: unique labels
    # y-axis: counts of each label
    # Set the axis labels and title of the plot
    sns.barplot(dict_counts, x=dict_counts.keys(), y= dict_counts.values(),
                 ax=axis).set(xlabel=x_axis, ylabel=y_axis, title=title) 
    
# =======================================================================================================

# For more information see the following articles

# 1. https://stackoverflow.com/questions/56518924/how-we-can-get-the-total-number-of-neurons-from-a-model-in-keras-is-there-any-f

# 2. https://keras.io/guides/sequential_model/

# 3. https://www.geeksforgeeks.org/accessing-intermediate-layer-outputs-in-keras/

def count_dead_neurons(dataset, model):
    """
    This Is an old version and NOT used

    It uses a sample from the given dataset to count the number of "dead neurones"
    (units that are not being used) in each convolutional layer of a given model.

    Args:
        dataset (tf.data.Dataset): A dataset containing image-label pairs.
        model (tf.keras.Model): The trained model to inspect.

     Returns:
        pd.DataFrame: A DataFrame listing each convolutional layer and the number of dead neurons                      
    
    """

    # Extract a single batch of input data from the dataset
    for item, target in dataset:
        data = item

     # Get the output tensors and names of all convolutional layers in the model
    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    outputs_name = [layer.name for layer in model.layers if "conv" in layer.name]

    # Create a model that outputs activations for all convolutional layers
    layer_output_model = Model(inputs=model.input, outputs=outputs)

    # Run the input data through the model to get activations
    layer_output = layer_output_model.predict(data)   

    data_list = []

    # Iterate over each layer's activation output
    for index, layer in enumerate(layer_output):

        # Sum activations across the channel axis to detect dead neurons
        mean_activations = np.sum(layer, axis=3) 
        
        # Total number of feature maps (neurons) in the layer
        total_activations = len(mean_activations)

        # Count how many feature maps are completely inactive 
        zero_count = np.count_nonzero(mean_activations==0)

        # Append results as [layer name, number of dead neurons]
        data_list.append([outputs_name[index], zero_count])
       
     # Convert the results to a pandas DataFrame
    data_df = pd.DataFrame(data_list, columns=('layer', 'Number of Dead neurons in %'))

    return data_df

# =======================================================================================================


def count_dead_neurons_v2(dataset, model):
    """
    This Is an old version and NOT used

    Finds the percentage of "dead neurones" (filters with no average activity)
    in each convolutional layer of a model using a dataset as input. A neurone (filter)
    is said to be "dead" if it has zero average activation across all sites and batches.

    Args:
        dataset (tf.data.Dataset): A dataset containing image-label pairs (batched).
        model (tf.keras.Model): The trained model whose convolutional layers will be analyzed.

    Returns:
        pd.DataFrame: A DataFrame with each convolutional layer's name and the number of dead neurons (filters).

    """ 

   
    # Get outputs and names of all convolutional layers
    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    outputs_name = [layer.name for layer in model.layers if "conv" in layer.name]

    # Create a model that outputs activations from all convolutional layers
    layer_output_model = Model(inputs=model.input, outputs=outputs)

    # To store [layer name, dead neuron count] for each conv layer
    data_list = []

    # Take only the first batch from the dataset
    for item, target in dataset.take(1):
       
        # Produces a list of the 3 convolutions layer.  Each layer has shape (Batch, Height, Width, Filter)
        layer_output = layer_output_model.predict(item, verbose=0)
        
        # Iterate through each convolutional layer's activation output
        for index, layer in enumerate(layer_output):
            
            # Averages the activation for each neuron in a layer
            # Mean activations is an array with length of the number of filters
            # in a convolutional layer
            mean_activations = np.mean(layer, axis=(0,1,2))

            # Count how many filters have zero mean activation (i.e., dead)
            total_activations = len(mean_activations)
            zero_count = np.count_nonzero(mean_activations==0)

            # accumulate the dead neuron count for each layer  
            if len(data_list) <= index:
                data_list.append([outputs_name[index], zero_count])
            else:
                data_list[index][1] += zero_count

    # Convert the result into a DataFrame
    data_df = pd.DataFrame(data_list, columns=('layer', 'Number of Dead neurons in %'))

    return data_df

# =======================================================================================================

  # Image example of 4d array
  # For more information:
  #     https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

def count_dead_neurons_v3(dataset, model):
    """

    Finds the proportion of "dead neurones" (filters that are not being used) in each convolutional layer
    across the whole dataset.  When all of its inputs fail to make it fire above a very small level,
    a neurone is said to be dead.
    
    Args:
        dataset (tf.data.Dataset): A batched dataset containing image-label pairs.
        model (tf.keras.Model): The trained model to evaluate.
    
    Returns:
        pd.DataFrame: A DataFrame with each convolutional layer and the percentage of dead neurons.

    """ 

   
    # Get the outputs and number of neurons (filters) for each convolutional layer
    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    max_neurons = [layer.output.shape[-1] for layer in model.layers if "conv" in layer.name]    
    outputs_name = [f'Conv2d_{i}' for i in range(len(outputs))]
    

    # Create a model that outputs activations from all convolutional layers
    layer_output_model = Model(inputs=model.input, outputs=outputs)

    # To store [layer name, dead neuron count] for each conv layer
    data_list = []

    # For more information: https://www.geeksforgeeks.org/initialize-dictionary-with-default-values/
    # Initialize a dictionary to track whether each neuron has ever been activated
    # False = dead; True = active
    neuron_count = {key: {f'neuron_{i+1}': False for i in range(max_neurons[index]) }
                     for index, key in enumerate(outputs_name)}

    # Loop over the dataset in batches
    for batch, _ in dataset:

       
        # Produces a list of the 3 convolutions layer.  Each layer has shape (Batch, Height, Width, Filter)
        layer_output = layer_output_model.predict(batch, verbose=0)

        # Loops through each layer in the layer_output
        # Calculates if the neuron has been activated
        for index, layer in enumerate(layer_output):
            
            # Averages the activation for each neuron in a layer
            # Mean activations is an array with length of the number of filters
            # in a convolutional layer
            max_activations = np.max(layer, axis=(0,1,2))
           
           # loops through the neurons in mean_activations
           # Checks to see if a neuron is greater than a very small number next to 0
           # If it is greater than 0 then updated the neuron count dictionary
           # entry for that neuron to True for active
            for neuron_index, neuron in enumerate(max_activations):
                # Small value to distinguish from exact zero
                if neuron >= 1e-09:

                    neuron_count[outputs_name[index]][f'neuron_{neuron_index+1}'] = True


    # Loop through the neuron_count dictionary
    # Find the total number of activations
    # Sum all the False values for non active neurons
    # Calculate the percentage
    for key, value in neuron_count.items():
        total_activations = len(value) 
        zero_count = sum(neuron_state == False for neuron_state in value.values())
        data_list.append([key, round(zero_count / total_activations  * 100, 2)])
        

    # Convert list to Dataframe
    data_df = pd.DataFrame(data_list, columns=('layer', 'Number of Dead neurons in %'))

    return data_df

# =======================================================================================================


def display_neurons_count(model_build, dataset, *args):
    """
    Plots the percentage of dead neurons in each convolutional layer of a model

    Args:
        model_build (tf.keras.Model): The model to evaluate.
        dataset (tf.data.Dataset): A dataset of input samples to use for analysis.
        *args (optional): A tuple containing:
            - label (str): Label to use for the plot legend.
            - ax (matplotlib.axes.Axes): Axis to plot on (enables subplotting or figure reuse).

    Output:
        Displays the result as a line plot using Seaborn
    """

    # Unpack optional label and axis if provided
    if args:
        label=args[0][0]
        ax=args[0][1]
    else:
        label=None
        ax=None 

    # Count dead neurons using version 3
    count = count_dead_neurons_v3(dataset, model_build )

     # Plot dead neuron percentages per layer
    sns.lineplot(data=count,
                 x='layer', y='Number of Dead neurons in %', label=label, ax=ax)
    
# =======================================================================================================

def display_model_times(time_history, reload=False):
    """
     Displays the total training time and average epoch duration for a model.

     Args:
        time_history (TimeHistory or list): Either a custom TimeHistory object with a `.times` attribute,
                                            or a list of epoch durations in seconds.
        reload (bool): Set to True if `time_history` is already a list of times (e.g., from saved logs).
    
    Output:
        Prints total training time and average time per epoch

    
    """
    
    # Get an object's list of times or use as-is if reloaded
    if not reload:
        time_history_result = time_history.times
    else: 
        time_history_result = time_history
    
    # Convert times to readable timedelta format
    model_average_time = timedelta(seconds=np.mean(time_history_result))
    model_total_time = timedelta(seconds=np.sum(time_history_result))

    # Print the total and average times
    print('Total Time:', str(model_total_time),
          'Average Epoch Time:', str(model_average_time))
    
# =======================================================================================================
    
def compare_times(*args, reload=False):
    """
    Compares the length of training for different models and displays in a summary table.

    Args:
        *args: One or more TimeHistory objects or lists of epoch durations (in seconds).
        reload (bool): Set to True if the input args are already lists of durations.

    
    """  

    # Initialize dictionary to hold comparison data
    data = {'Average Epoch': [],
            'Total Time': [],
            'Model':[]}

    # Process each model's time history
    for i, arg in enumerate(args):
         # Extract times if TimeHistory object
        if not reload:
            arg = arg.times
        else:
            arg

        time_history_result = arg
        data['Average Epoch'].append(str(timedelta(seconds=np.mean(time_history_result))))
        data['Total Time'].append(str(timedelta(seconds=np.sum(time_history_result))))
        data['Model'].append(f'Model_{i+1}')

    # Create and return a DataFrame for side-by-side comparison
    df = pd.DataFrame(data).set_index('Model')
    
    return df

# =======================================================================================================

# Image example of 4d array
# For more information:
#   https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

    
def mean_activation(dataset, model):
    """
    Calculates the mean activation of each neuron (filter) in all convolutional layers
    of a model across the entire dataset
    

    Args:
        dataset (tf.data.Dataset): A batched dataset containing input images.
        model (tf.keras.Model): The trained model to analyze.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a convolutional layer,
                    and each row corresponds to a neuron's average activation across all samples.

    """
   
    # Get the output tensors of all convolutional layers
    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    max_neurons = [layer.output.shape[-1] for layer in model.layers if "conv" in layer.name]
    outputs_name = [layer.name for layer in model.layers if "conv" in layer.name]

    # Create a model that outputs activations from all convolutional layers
    layer_output_model = Model(inputs=model.input, outputs=outputs)

    # To store [layer name, dead neuron count] for each conv layer
    data_list = []

    
    # For more information: https://www.geeksforgeeks.org/initialize-dictionary-with-default-values/
    # Initialize a dictionary to track whether each neuron has ever been activated
    # False = dead; True = active
    neuron_count = {key: {f'neuron_{i+1}': False for i in range(max_neurons[index]) }
                     for index, key in enumerate(outputs_name)}

    total_samples = 0

    # Iterate over all batches in the dataset
    for batch, _ in dataset:

        batch_size = batch.get_shape()[0]
        total_samples += batch_size

       
        # Produces a list of the 3 convolutions layer.  Each layer has shape (Batch, Height, Width, Filter)
        # Get activations for the current batch
        layer_output = layer_output_model.predict(batch, verbose=0)
    
        # Loops through each layer in the layer_output
        # Calculates if the neuron has been activated
        for index, layer in enumerate(layer_output):
            
            # Averages the activation for each neuron in a layer
            # Mean activations is an array with length of the number of filters
            # in a convolutional layer
            mean_activations = np.mean(layer, axis=(0,1,2))
     
            # For more information: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
            # Multiply by batch size to prepare for weighted average
            mean_activations = mean_activations * batch_size
  
            # Accumulate activation values across batches
            if len(data_list) <= index:
                data_list.append(mean_activations)
            else:
                data_list[index] += mean_activations
                
    # Finalize the average by dividing by total number of samples
    data_list = [arr / total_samples for arr in data_list]

     # Format results into a DataFrame: one column per layer, one row per neuron
    mean_activations_df = pd.DataFrame(data_list, index=[f'layer_{i+1}' for i in range(len(outputs_name))]).transpose()

    return mean_activations_df

# =======================================================================================================

def display_mean_activations(model_build, dataset, layer, bin_start=0, bin_stop=0.5, bin_num=11, *args):
    """
     Plots a histogram of mean activations for a specific convolutional layer in a model.

    Args:
        model_build (tf.keras.Model): The trained model to evaluate.
        dataset (tf.data.Dataset): A batched dataset of input samples.
        layer (int): The index (1-based) of the convolutional layer to visualize.
        bin_start (float): The start of the bin range for the histogram.
        bin_stop (float): The end of the bin range for the histogram.
        bin_num (int): Number of bins to divide the range into.
        *args (optional): Tuple containing:
            - label (str): Label for the plot legend.
            - ax (matplotlib.axes.Axes): Axis to plot on (for subplotting support).

    Output:
        Displays a histogram showing the distribution of mean activations across filters
    
    """
    # Unpack optional plotting arguments if provided
    if args:

        label=args[0][0]
        ax=args[0][1]
        axis_label=ax.set_xlabel  
        axis_legend=ax  

    else:

        label=None
        ax=None
        axis_label=plt.xlabel
        axis_legend=plt

     # Compute mean activations for all conv layers
    count = mean_activation(dataset, model_build )

    # Define histogram bins
    bins = np.linspace(bin_start, bin_stop, bin_num)

    # Plot the histogram of mean activations for the specified layer
    sns.histplot(data=count, x=f'layer_{layer}', stat='percent', bins=bins, label=label, alpha=0.4, ax=ax )

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Label the x-axis and show the legen
    axis_label(f'layer_{layer} Mean Activation')
    axis_legend.legend(fontsize='medium')
 