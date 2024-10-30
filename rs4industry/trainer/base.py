from collections import defaultdict
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from rs4industry.model.base import BaseModel
from rs4industry.model.retriever import BaseRetriever
from rs4industry.utils import get_logger, batch_to_device
from rs4industry.eval import get_eval_metrics
from rs4industry.config import TrainingArguments
from rs4industry.callbacks import Callback, EarlyStopCallback, CheckpointCallback


class Trainer(object):
    def __init__(self, model, config, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.config: TrainingArguments = self.load_config(config)
        self._check_checkpoint_dir()
        self.model: BaseModel = model.to(self.config.device)
        self.optimizer = self.get_optimizer(
            self.config.optimizer,
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay    
        )
        self.lr_scheduler = self.get_lr_scheduler()
        self.logger = get_logger(self.config)
        self.global_step = 0    # global step counter, load from checkpoint if exists
        self.cur_global_step = 0    # current global step counter, record the steps of this training
        self._last_eval_epoch = -1
        self.callbacks: List[Callback] = self.get_callbacks()
        self._total_train_samples = 0
        self._total_eval_samples = 0

    def load_config(self, config: Union[Dict, str]) -> TrainingArguments:
        if isinstance(config, TrainingArguments):
            return config
        if isinstance(config, dict):
            config_dict = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return TrainingArguments.from_dict(config_dict)

    def get_callbacks(self):
        callbacks = []
        if self.config.earlystop_metric is not None:
            callbacks.append(EarlyStopCallback(
                monitor_metric=self.config.earlystop_metric,
                strategy=self.config.earlystop_strategy,
                patience=self.config.earlystop_patience,
                maximum="max" in self.config.earlystop_metric_mode,
                save=self.config.checkpoint_best_ckpt,
                model=self.model,
                checkpoint_dir=self.config.checkpoint_dir
            ))
        if self.config.checkpoint_steps is not None:
            callbacks.append(CheckpointCallback(
                model=self.model,
                step_interval=self.config.checkpoint_steps,
                checkpoint_dir=self.config.checkpoint_dir
            ))
        return callbacks
        
    

    def get_train_loader(self, train_dataset: Optional[Union[Dataset, str]]=None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size)
        return loader
    
    def get_eval_loader(self, eval_dataset: Optional[Union[Dataset, str]]=None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        loader = DataLoader(eval_dataset, batch_size=self.config.eval_batch_size)
        return loader
    
    def get_item_loader(self, eval_dataset: Optional[Union[Dataset, str]]=None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        loader = DataLoader(eval_dataset, batch_size=self.config.item_batch_size)
        return loader

    def fit(self, train_dataset, eval_dataset=None, *args, **kwargs):
        train_loader = self.get_train_loader(train_dataset)
        item_loader = self.get_item_loader(train_dataset.item_feat_dataset) if train_dataset.item_feat_dataset is not None else None
        if eval_dataset is not None:
            eval_loader = self.get_eval_loader(eval_dataset)
        else:
            eval_loader = None

        for callback in self.callbacks:
            callback.on_train_begin(train_dataset, eval_dataset, *args, **kwargs)
        
        stop_training = False
        try:
            for epoch in range(self.config.epochs):
                epoch_total_loss = 0.0
                epoch_total_bs = 0
                self.logger.info(f"Start training epoch {epoch}")
                self.model.train()

                for step, batch in enumerate(train_loader):
                    if (eval_loader is not None) and self._check_if_eval(epoch, step):
                        stop_training = self.evaluation_loop(eval_loader, item_loader, epoch)

                    batch = batch_to_device(batch, self.config.device) # move batch to GPU if available
                    batch_size = batch[list(batch.keys())[0]].shape[0]
                    loss_dict = self._train_batch(batch, item_loader=item_loader, *args, **kwargs)
                    loss = loss_dict['loss']

                    loss.backward()
                    # gradient accumulation
                    if self.config.gradient_accumulation_steps is None or self.config.gradient_accumulation_steps == 1:
                        self.gradient_clipping(self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        if (self.global_step+1) % self.config.gradient_accumulation_steps == 0:
                            self.gradient_clipping(self.config.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    epoch_total_loss += loss.item() * batch_size
                    epoch_total_bs += batch_size
                    if self.global_step % self.config.logging_steps == 0:
                        mean_total_loss = epoch_total_loss / epoch_total_bs
                        self.logger.info(f"Epoch {epoch}/{self.config.epochs} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                        if len(loss_dict) > 1:
                            self.logger.info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))


                    for callback in self.callbacks:
                        callback.on_batch_end(
                            epoch = epoch,
                            step = self.global_step,
                            logs = loss_dict,
                            item_loader = item_loader   # for retriever model to update item vectors when saving
                        )
                
                    self.global_step += 1
                    self.cur_global_step += 1

                    if stop_training:
                        self.logger.info("[Earlystop] Stop training at epoch {}, {} global steps:".format(epoch, self.config.epochs, self.global_step))
                        break
                    # except KeyboardInterrupt:
                    #     self.logger.info(f"[KeyboardInterrupt] Stop training at {self.global_step} steps")
                    #     stop_training = True
                    #     break

                for callback in self.callbacks:
                        callback.on_epoch_end(epoch, self.global_step, *args, **kwargs)

                if stop_training:
                    break

                self._total_train_samples = epoch_total_bs
                self.logger.info(f"[Finished] Stop training at {self.global_step} steps")
        
        except KeyboardInterrupt:
            self.logger.info(f"[KeyboardInterrupt] Stop training at {self.global_step} steps")

        for callback in self.callbacks:
            callback.on_train_end(checkpoint_dir=self.config.checkpoint_dir, *args, **kwargs)

        if self.model.model_type == "retriever":
            # update all item vectors
            self.model.update_item_vectors(item_loader)

        if eval_loader is not None:
            self.logger.info("Start final evaluation...")
            self.evaluation_loop(eval_loader, item_loader, epoch)
        
        self.save_state(self.config.checkpoint_dir)


    def evaluation_loop(self, eval_loader, item_loader, epoch, *args, **kwargs):
        stop_training = False
        self.logger.info("Start evaluation...")
        self.model.eval()
        if self.model.model_type == "retriever":
            self.model.update_item_vectors(item_loader)
        eval_outputs = []
        eval_total_bs = 0
        for eval_step, eval_batch in enumerate(eval_loader):
            eval_batch = batch_to_device(eval_batch, self.config.device) # move batch to GPU if available, otherwise do nothing
            eval_batch_size = eval_batch[list(eval_batch.keys())[0]].shape[0]
            metrics = self._eval_batch(eval_batch, *args, **kwargs)
            eval_outputs.append((metrics, eval_batch_size))
            eval_total_bs += eval_batch_size
        metrics = self.eval_epoch_end(eval_outputs)
        self._total_eval_samples = eval_total_bs
        self.logger.info("Validation at Epoch {} Step {}:".format(epoch, self.global_step))
        self.log_dict(metrics)

        for callback in self.callbacks:
            if isinstance(callback, EarlyStopCallback):
                stop_training = callback.on_eval_end(epoch, self.global_step, metrics, *args, **kwargs)
            else:
                callback.on_eval_end(epoch, self.global_step, *args, **kwargs)

        return stop_training

    def _check_if_eval(self, epoch, step):
        if self.config.evaluation_strategy == 'epoch':
            if (epoch % self.config.eval_interval == 0) and (self._last_eval_epoch != epoch):
                self._last_eval_epoch = epoch
                return True
            return False
        elif self.config.evaluation_strategy == 'step':
            if self.global_step % self.config.eval_interval == 0:
                return True
            return False
        else:
            raise ValueError(f'Unknown evaluation strategy: {self.config.evaluation_strategy}')
    

    def _train_batch(self, batch, *args, **kwargs):
        loss_dict = self.model.cal_loss(batch=batch, *args, **kwargs)
        return loss_dict


    def _eval_batch(self, batch, *args, **kwargs) -> Dict:
        """ Evaluate the model on a batch, return metrics.

        Args:
            batch (Dict): The input batch.

        Returns:
            Dict: The metrics.
        """
        with torch.no_grad():
            self.model.eval()
            k = max(self.config.cutoffs) if self.config.cutoffs is not None else None
            outputs = self.model.eval_step(batch, k=k, *args, **kwargs)
            metrics: dict = self.compute_metrics(outputs)
            return metrics
    

    @torch.no_grad()
    def eval_epoch_end(self, outputs: List[Tuple]) -> Dict:
        """ Aggregate the metrics from the evaluation batch.

        Args:
            outputs (List): The output of the evaluation batch. It is a list of tuples, 
                where the first element is the metrics (Dict) and the second element is the batch size.

        Returns:
            Dict: The aggregated metrics.
        """
        if self.model.model_type == "retriever":
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = (metric * bs).sum() / bs.sum()
            return out
        else:
            # ranker: AUC, Logloss
            out = {}
            output, bs = zip(*outputs)
            pred, target = zip(*output)
            pred = torch.cat(pred, dim=-1)
            target = torch.cat(target, dim=-1)
            metrics: list = get_eval_metrics(self.config.metrics, self.model.model_type)
            for metric, func in metrics:
                out[metric] = func(pred, target)
            return out
            

    @torch.no_grad()
    def compute_metrics(self, output: Tuple):
        """ Compute the metrics given the output of the model.

        Args:
            output (Tuple): The output of the model.

        Returns:
            Dict: The computed metrics.
        """
        model_type = "retriever" if isinstance(self.model, BaseRetriever) else "ranker"
        metrics: list = get_eval_metrics(self.config.metrics, model_type)
        cutoffs = self.config.cutoffs
        output_dict = {}
        if model_type == "retriever":
            for metric, func in metrics:
                for cutoff in cutoffs:
                    output_dict[f"{metric}@{cutoff}"] = func(*output, cutoff)
        else:
            output_dict = (output[0].cpu(), output[1].cpu())    # (pred, target)
        return output_dict


    def log_dict(self, d: Dict):
        """Log a dictionary of values."""
        output_list = [f"{k}={v}" for k, v in d.items()]
        self.logger.info(", ".join(output_list))


    def get_optimizer(self, name, params, lr, weight_decay):
        r"""Return optimizer for specific parameters.

        .. note::
            If no learner is assigned in the configuration file, then ``Adam`` will be used.

        Args:
            params: the parameters to be optimized.

        Returns:
            torch.optim.optimizer: optimizer according to the config.
        """
        learning_rate = lr
        decay = weight_decay
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def get_lr_scheduler(self):
        return None


    def save_state(self, checkpoint_dir):
        """Save the state of the trainer."""
        # save the parameters of the model
        # save model configuration, enabling loading the model later
        self.model.save(checkpoint_dir)

        # save the optimizer state and the scheduler state
        optimizer_state = {"optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            optimizer_state["scheduler"] = self.scheduler.state_dict()
        torch.save(optimizer_state, os.path.join(checkpoint_dir, 'optimizer_state.pt'))

        # save the trainer configurations
        with open(os.path.join(checkpoint_dir, 'trainer_config.json'), 'w') as fp:
            json.dump(self.config.to_dict(), fp, indent=4)

        # save the trainer state
        with open(os.path.join(checkpoint_dir, 'trainer_state.json'), 'w') as fp:
            json.dump(self.state, fp, indent=4)
        self.logger.info(f"Saved the model and trainer state to {checkpoint_dir}.")


    @property
    def state(self):
        state_dict = {
            "global_step": self.global_step,
        }
        return state_dict


    def _check_checkpoint_dir(self):
        checkpoint_dir = self.config.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("Checkpoint directory must be specified.")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            # check if the checkpoint_dir is empty
            if len(os.listdir(checkpoint_dir)) == 0:
                pass
            else:
                raise ValueError(f"Checkpoint directory '{checkpoint_dir}' is not empty.")

    def gradient_clipping(self, clip_norm):
        if (clip_norm is not None) and (clip_norm > 0):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

