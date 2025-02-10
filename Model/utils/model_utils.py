
import numpy as np

def save_model_data(path_str,
                    model_build, model_build_str,
                    model_history, model_history_str,
                    model_time, model_time_str):
    """
    Saves model-related data (history, model, and time) to specified paths.

    Parameters:
    - path_str (str): Base directory for saving the files.
    - model_build (tf.keras.Model): The built model to save.
    - model_build_str (str): Identifier string for the model file.
    - model_history (History): Training history object to save.
    - model_history_str (str): Identifier string for the history file.
    - model_time (Any): Object holding timing data to save.
    - model_time_str (str): Identifier string for the timing data file.
    """
     # Save model history
    np.save(path_str + model_history_str + '.npy', model_history.history)

    # Save the model
    model_build.save(path_str + model_build_str + '.keras')

    # Save timing data
    np.save(path_str + model_time_str + '.npy', model_time.times)