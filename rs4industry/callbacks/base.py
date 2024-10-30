class Callback(object):
    def __init__(self):
        pass

    def on_train_begin(self, logs={}, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, step, logs={}, *args, **kwargs):
        pass

    def on_batch_end(self, epoch, step, logs={}, *args, **kwargs):
        pass

    def on_eval_end(self, epoch, step, logs={}, *args, **kwargs):
        pass

    def on_train_end(self, epoch, step, logs={}, *args, **kwargs):
        pass
