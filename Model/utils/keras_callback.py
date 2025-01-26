import time
import keras


class TimeHistory(keras.callbacks.Callback):
    """
    https://stackoverflow.com/a/43186440 Author:Marcin Możejko

    Track Computation of Models

    Example:

    time_callback = TimeHistory()
    model.fit(..., callbacks=[..., time_callback],...)
    times = time_callback.times 
    
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)