from dataclasses import dataclass

@dataclass
class CallbackOutput:
    save_checkpoint: str = None
    stop_training: bool = False


class Callback(object):
    def __init__(self):
        pass

    def on_train_begin(self, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_epoch_end(self, epoch, step, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_batch_end(self, epoch, step, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_eval_end(self, epoch, step, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_train_end(self, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()