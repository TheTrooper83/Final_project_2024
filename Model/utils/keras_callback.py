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


class CLR_history(keras.callbacks.Callback):
  
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []


    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
    
    # def on_epoch_end(self, epoch, logs={}):
    #     self.losses.append(logs.get('loss'))
    #     self.accuracy.append(logs.get('accuracy'))


    # def on_test_batch_end(self, batch, logs={}):        
    #     self.val_losses.append(logs.get('val_loss'))        
    #     self.val_accuracy.append(logs.get('val_accuracy'))

    # def on_epoch_end(self, batch, logs={}):
    #     self.lr_rate.append(self.model.optimizer.lr)
    
class stop_model(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch + 1

        if self.epoch == 25:
            self.model.stop_training = True
              