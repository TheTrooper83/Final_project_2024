import time
import keras

# ==========================================================================================

class TimeHistory(keras.callbacks.Callback):
    """
    https://stackoverflow.com/a/43186440 Author:Marcin Możejko

    Track Computation of Models

    Example:

    time_callback = TimeHistory()
    model.fit(..., callbacks=[..., time_callback],...)
    times = time_callback.times 
    
    """
    # Called at the start of training
    def on_train_begin(self, logs={}):
         # Initialize a list to store epoch durations
        self.times = []

    # Called at the beginning of each epoch
    def on_epoch_begin(self, batch, logs={}):
         # Record the current time to measure the epoch's duration
        self.epoch_time_start = time.time()

    # Called at the end of each epoch
    def on_epoch_end(self, batch, logs={}):
        # Calculate and store the time taken for the epoch
        self.times.append(time.time() - self.epoch_time_start)

# ==========================================================================================

class CLR_history(keras.callbacks.Callback):
    """
    Custom Keras callback for tracking loss and accuracy at the end of each batch
    during training, used alongside Cyclical Learning Rate (CLR) schedules.

    Attributes:
        losses (list): Stores loss values after each batch.
        accuracy (list): Stores accuracy values after each batch.
    """
    
    # Called at the start of training
    def on_train_begin(self, logs={}):
        # Initialize a list to store loss and accuracy values.
        self.losses = []
        self.accuracy = []

    #  Called at the end of each batch
    def on_batch_end(self, batch, logs={}):
        # Append Loss and accuracy to to the list
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
 # ==========================================================================================
    
class stop_model(keras.callbacks.Callback):
    """
    Custom Keras callback to stop training early after a fixed number of epochs.  

    Attributes:
        epoch (int): Tracks the current epoch number.
    """

    # Called at the end of each epoch. Stops training when epoch 25 is reached.
    def on_epoch_end(self, epoch, logs={}):
        # Convert to 1-based index
        self.epoch = epoch + 1

        # Stop training once 25 epochs have completed
        if self.epoch == 25:
            self.model.stop_training = True
              