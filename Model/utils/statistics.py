from keras.models import  Model
import pandas as pd
import numpy as np
import tensorflow as tf
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
            
            # print(layer.shape)
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