from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import  Model
import pandas as pd
import numpy as np
import tensorflow as tf

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




# https://stackoverflow.com/questions/56518924/how-we-can-get-the-total-number-of-neurons-from-a-model-in-keras-is-there-any-f

# https://keras.io/guides/sequential_model/

# https://www.geeksforgeeks.org/accessing-intermediate-layer-outputs-in-keras/

def count_dead_neurons(dataset, model):

    for item, target in dataset:
        data = item

    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    outputs_name = [layer.name for layer in model.layers if "conv" in layer.name]

    # Create a new model to get the output of the first layer
    # layer_output_model = Model(inputs=simple_relu_model_best.input, outputs=simple_relu_model_best.layers[1].output)
    layer_output_model = Model(inputs=model.input, outputs=outputs)

    # Get the output of the first layer
    # layer_output = layer_output_model.predict(sample_data)
    layer_output = layer_output_model.predict(data)

    # Print the output of the first layer
    # print("Output of the first layer:\n", layer_output)

    data_list = []
    for index, layer in enumerate(layer_output):

        # mean_activations = np.mean(layer, axis=(0,1))
        mean_activations = np.sum(layer, axis=3)
        print(mean_activations[0])

        # zero_count = 0
        # total_activations = layer.size
        total_activations = len(mean_activations)
        zero_count = np.count_nonzero(mean_activations==0)
        # data_list.append([outputs_name[index], round(zero_count / total_activations  * 100, 2)])
        data_list.append([outputs_name[index], zero_count])
        print('test')

    data_df = pd.DataFrame(data_list, columns=('layer', 'Number of Dead neurons in %'))

    return data_df


def count_dead_neurons_v2(dataset, model):
    
    

   

    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    outputs_name = [layer.name for layer in model.layers if "conv" in layer.name]

    # Create a new model to get the output of the first layer
    # layer_output_model = Model(inputs=simple_relu_model_best.input, outputs=simple_relu_model_best.layers[1].output)
    layer_output_model = Model(inputs=model.input, outputs=outputs)

    data_list = []
    # activations = []

    for item, target in dataset.take(1):
        # data = item
       
        # Produces a list of the 3 convolutions layer.  Each layer has shape (Batch, Height, Width, Filter)
        layer_output = layer_output_model.predict(item, verbose=0)
        # print(layer_output[0].shape)

        # Print the output of the first layer
        # print("Output of the first layer:\n", layer_output)

        
        for index, layer in enumerate(layer_output):
            
            # Averages the activation for each neuron in a layer
            # Mean activations is an array with length of the number of filters
            # in a convolutional layer
            mean_activations = np.mean(layer, axis=(0,1,2))


            # print(mean_activations)
        # activations.append(layer_output)
   
    # print(activations[0][0].shape)
    # activations = np.concatenate(activations, axis=0)
    # print(activations.shape)

    #         # zero_count = 0
            # total_activations = layer.size
            total_activations = len(mean_activations)
            zero_count = np.count_nonzero(mean_activations==0)
            # data_list.append([outputs_name[index], round(zero_count / total_activations  * 100, 2)])
            if len(data_list) <= index:

                data_list.append([outputs_name[index], zero_count])
            else:
                data_list[index][1] += zero_count
    #         print('test')
    # print(len(activations))

    data_df = pd.DataFrame(data_list, columns=('layer', 'Number of Dead neurons in %'))

    return data_df

def count_dead_neurons_v3(dataset, model):

    # Image example of 4d array
    # https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

   

    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    max_neurons = [layer.output.shape[-1] for layer in model.layers if "conv" in layer.name]    
    # outputs_name = [f'Conv2d_{i}' for i,layer in enumerate(model.layers) if "conv" in layer.name]
    outputs_name = [f'Conv2d_{i}' for i in range(len(outputs))]
    

    # Create a new model to get the output of the first layer

    layer_output_model = Model(inputs=model.input, outputs=outputs)

    data_list = []

    # https://www.geeksforgeeks.org/initialize-dictionary-with-default-values/
    # Initialise Dictionary with default values
    # Initializes a dictionary with false values for each neuron in a layer
    neuron_count = {key: {f'neuron_{i+1}': False for i in range(max_neurons[index]) }
                     for index, key in enumerate(outputs_name)}


    for batch, _ in dataset:

       
        # Produces a list of the 3 convolutions layer.  Each layer has shape (Batch, Height, Width, Filter)
        layer_output = layer_output_model.predict(batch, verbose=0)

        # loops through each layer in the layer_output
        # calculates if the neuron has been activated
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
                if neuron >= 1e-09:

                    neuron_count[outputs_name[index]][f'neuron_{neuron_index+1}'] = True


    # loop through the neuron_count dictionary
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


def display_neurons_count(model_build, dataset, *args):

    if args:

        label=args[0][0]
        ax=args[0][1]

    else:

        label=None
        ax=None 

    count = count_dead_neurons_v3(dataset, model_build )

    sns.lineplot(data=count,
                 x='layer', y='Number of Dead neurons in %', label=label, ax=ax)




def display_model_times(time_history, reload=False):
    
    if not reload:
        time_history_result = time_history.times
    else: 
        time_history_result = time_history
    model_average_time = timedelta(seconds=np.mean(time_history_result))
    model_total_time = timedelta(seconds=np.sum(time_history_result))
    print('Total Time:', str(model_total_time),
          'Average Epoch Time:', str(model_average_time))
    
def compare_times(*args, reload=False):  

    data = {'Average Epoch': [],
            'Total Time': [],
            'Model':[]}

    for i, arg in enumerate(args):

        if not reload:
            arg = arg.times
        else:
            arg

        time_history_result = arg
        data['Average Epoch'].append(str(timedelta(seconds=np.mean(time_history_result))))
        data['Total Time'].append(str(timedelta(seconds=np.sum(time_history_result))))
        data['Model'].append(f'Model_{i+1}')


    df = pd.DataFrame(data).set_index('Model')
    
    return df
    

    
def mean_activation(dataset, model):

    # Image example of 4d array
    # https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

   

    outputs = [layer.output for layer in model.layers if "conv" in layer.name]
    max_neurons = [layer.output.shape[-1] for layer in model.layers if "conv" in layer.name]
    outputs_name = [layer.name for layer in model.layers if "conv" in layer.name]

    # Create a new model to get the output of the first layer

    layer_output_model = Model(inputs=model.input, outputs=outputs)

    data_list = []

    
    # https://www.geeksforgeeks.org/initialize-dictionary-with-default-values/
    # Initialise Dictionary with default values
    # Initializes a dictionary with false values for each neuron in a layer
    neuron_count = {key: {f'neuron_{i+1}': False for i in range(max_neurons[index]) }
                     for index, key in enumerate(outputs_name)}

    total_samples = 0
    for batch, _ in dataset:
        batch_size = batch.get_shape()[0]
        # print(batch[0])
        total_samples += batch_size

       
        # Produces a list of the 3 convolutions layer.  Each layer has shape (Batch, Height, Width, Filter)
        layer_output = layer_output_model.predict(batch, verbose=0)
    
        # loops through each layer in the layer_output
        # calculates if the neuron has been activated
        for index, layer in enumerate(layer_output):
            
            # Averages the activation for each neuron in a layer
            # Mean activations is an array with length of the number of filters
            # in a convolutional layer
            mean_activations = np.mean(layer, axis=(0,1,2))
     
            # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
            mean_activations = mean_activations * batch_size
  

            if len(data_list) <= index:
                data_list.append(mean_activations)
            else:
                data_list[index] += mean_activations
                
    
    data_list = [arr / total_samples for arr in data_list]
    mean_activations_df = pd.DataFrame(data_list, index=[f'layer_{i+1}' for i in range(len(outputs_name))]).transpose()

    return mean_activations_df

def display_mean_activations(model_build, dataset, layer, bin_start=0, bin_stop=0.5, bin_num=11, *args):

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

    count = mean_activation(dataset, model_build )

    bins = np.linspace(bin_start, bin_stop, bin_num)

      
    sns.histplot(data=count, x=f'layer_{layer}', stat='percent', bins=bins, label=label, alpha=0.4, ax=ax )
    plt.xticks(rotation=45)
    axis_label(f'layer_{layer} Mean Activation')
    axis_legend.legend(fontsize='medium')
 