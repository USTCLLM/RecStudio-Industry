import os


from rs4industry.callbacks.base import Callback


class CheckpointCallback(Callback):
    def __init__(self, model, step_interval: int, checkpoint_dir: str, **kwargs):
        """ CheckpointCallback, saves model checkpoints at a given step interval.

        Args:
            model (nn.Module): Model to save checkpoints for.
            step_interval (int): Interval at which to save checkpoints.
            checkpoint_dir (str): Directory to save checkpoints in.
        """
        super().__init__(**kwargs)
        self.model = model
        self.step_interval = step_interval
        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint_step = 0

    
    def on_batch_end(self, epoch, step, logs=..., *args, **kwargs):
        if step > 0 and self.step_interval is not None:
            if (step - self.last_checkpoint_step) % self.step_interval == 0:
                self.save_checkpoint(step, item_loader=kwargs.get('item_loader', None))
                self.last_checkpoint_step = step

    def on_epoch_end(self, epoch, step, item_loader=None, *args, **kwargs):
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint-epoch-{epoch}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model.save(checkpoint_dir, item_loader=item_loader)
        print(f"Save checkpoint at epoch {epoch} into directory {checkpoint_dir}")

        
    def save_checkpoint(self, step: int, item_loader=None):
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model.save(checkpoint_dir, item_loader=item_loader)
        print(f"Save checkpoint at step {step} into directory {checkpoint_dir}")