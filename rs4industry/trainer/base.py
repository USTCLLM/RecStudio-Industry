from collections import defaultdict
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from rs4industry.utils import get_logger
from rs4industry.eval import get_eval_metrics
from rs4industry.config import TrainingArguments
from rs4industry.callbacks import Callback, EarlyStopCallback, CheckpointCallback


class Trainer(object):
    def __init__(self, model, config=None, train=True, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.config: TrainingArguments = self.load_config(config)
        self.train_mode = train
        self.model_type = model.model_type
        self._check_checkpoint_dir()

        self.accelerator = Accelerator()
        if self.accelerator.is_main_process:
            print(model)
        model.to(self.accelerator.device)
        optimizer = self.get_optimizer(
            self.config.optimizer,
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay    
        )
        self.model = self.accelerator.prepare(model)
        self.optimizer = self.accelerator.prepare(optimizer)
        self.lr_scheduler = self.get_lr_scheduler()
        self.logger = get_logger(self.config)
        self.global_step = 0    # global step counter, load from checkpoint if exists
        self.cur_global_step = 0    # current global step counter, record the steps of this training
        self._last_eval_epoch = -1
        if self.train_mode:
            self.callbacks: List[Callback] = self.get_callbacks()
        self._total_train_samples = 0
        self._total_eval_samples = 0
        self.item_vectors = None
        self.item_ids = None

    def load_config(self, config: Union[Dict, str]) -> TrainingArguments:
        if config is None:
            return TrainingArguments()
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
                checkpoint_dir=self.config.checkpoint_dir,
                is_main_process=self.accelerator.is_main_process
            ))
        if self.config.checkpoint_steps is not None:
            callbacks.append(CheckpointCallback(
                step_interval=self.config.checkpoint_steps,
                checkpoint_dir=self.config.checkpoint_dir,
                is_main_process=self.accelerator.is_main_process
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
        train_loader = self.accelerator.prepare(train_loader)

        if train_dataset.item_feat_dataset is not None:
            item_loader = self.accelerator.prepare(self.get_item_loader(train_dataset.item_feat_dataset))
        else:
            item_loader = None
        
        if eval_dataset is not None:
            eval_loader = self.get_eval_loader(eval_dataset)
            eval_loader = self.accelerator.prepare(eval_loader)
        else:
            eval_loader = None

        for callback in self.callbacks:
            callback_output = callback.on_train_begin(train_dataset, eval_dataset, *args, **kwargs)
            if callback_output.save_checkpoint is not None:
                self.save_state(callback_output.save_checkpoint)
        
        stop_training = False
        try:
            for epoch in range(self.config.epochs):
                epoch_total_loss = 0.0
                epoch_total_bs = 0
                self.model.train()

                self.log_info(f"Start training epoch {epoch}")
                for step, batch in enumerate(train_loader):
    
                    # check if need to do evaluation
                    if (eval_loader is not None) and self._check_if_eval(epoch, self.global_step):
                        metrics = self.evaluation_loop(eval_loader, item_loader)
                        self.log_info("Evaluation at Epoch {} Step {}:".format(epoch, self.global_step))
                        self.log_dict(metrics)

                        for callback in self.callbacks:
                            callback_output = callback.on_eval_end(epoch, self.global_step, metrics, *args, **kwargs)
                            if callback_output.save_checkpoint is not None:
                                self.save_state(callback_output.save_checkpoint)
                            if callback_output.stop_training and (not stop_training):
                                stop_training = True

                    # train one batch
                    batch_size = batch[list(batch.keys())[0]].shape[0]
                    loss_dict = self._train_batch(batch, item_loader=item_loader, *args, **kwargs)
                    loss = loss_dict['loss']

                    self.accelerator.backward(loss)

                    # gradient accumulation and gradient clipping by norm
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
                    if (self.global_step % self.config.logging_steps == 0):
                        mean_total_loss = epoch_total_loss / epoch_total_bs
                        self.log_info(f"Epoch {epoch}/{self.config.epochs-1} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                        if (len(loss_dict) > 1):
                            self.log_info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))

                    for callback in self.callbacks:
                        callback_output = callback.on_batch_end(
                            epoch = epoch,
                            step = self.global_step,
                            logs = loss_dict
                        )
                        if callback_output.save_checkpoint is not None:
                            self.save_state(callback_output.save_checkpoint)
                        if callback_output.stop_training and (not stop_training):
                            stop_training = True
                
                    self.global_step += 1
                    self.cur_global_step += 1

                    if stop_training:
                        self.log_info("[Earlystop] Stop training at epoch {}, {} global steps:".format(epoch, self.global_step))
                        break

                # print loss info at the end of each epoch
                mean_total_loss = epoch_total_loss / epoch_total_bs
                self.log_info(f"Epoch {epoch}/{self.config.epochs} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                if len(loss_dict) > 1:
                    self.log_info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))
                
                for callback in self.callbacks:
                    callback_output = callback.on_epoch_end(epoch, self.global_step, *args, **kwargs)
                    if callback_output.save_checkpoint is not None:
                        self.save_state(callback_output.save_checkpoint)
                    if callback_output.stop_training and (not stop_training):
                        stop_training = True

                if stop_training:
                    break

                self._total_train_samples = epoch_total_bs
        
        except KeyboardInterrupt:
            self.log_info(f"[KeyboardInterrupt] Stop training at {self.global_step} steps")

        self.log_info(f"[Finished] Stop training at {self.global_step} steps")

        for callback in self.callbacks:
            callback_output = callback.on_train_end(checkpoint_dir=self.config.checkpoint_dir, *args, **kwargs)
            if callback_output.save_checkpoint is not None:
                self.save_state(callback_output.save_checkpoint)

        if eval_loader is not None:
            self.log_info("Start final evaluation...")
            metrics = self.evaluation_loop(eval_loader, item_loader)
            self.log_info("Evaluation result:")
            self.log_dict(metrics)
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopCallback):
                    stop_training = callback.on_eval_end(epoch, self.global_step, metrics, *args, **kwargs)
                else:
                    callback.on_eval_end(epoch, self.global_step, *args, **kwargs)
        
        self.save_state(self.config.checkpoint_dir)


    @torch.no_grad()
    def evaluation(self, eval_dataset, *args, **kwargs):
        self.log_info("Start evaluation...")
        self.model.eval()
        item_loader = self.get_item_loader(eval_dataset.item_feat_dataset) if eval_dataset.item_feat_dataset is not None else None
        eval_loader = self.get_eval_loader(eval_dataset)
        eval_loader = self.get_eval_loader(eval_dataset)
        metrics = self.evaluation_loop(eval_loader, item_loader)
        self.log_info("Evaluation result:")
        self.log_dict(metrics)


    @torch.no_grad()
    def evaluation_loop(self, eval_loader, item_loader, *args, **kwargs) -> Dict:
        self.model.eval()
        if self.model_type == "retriever":
            self.item_vectors, self.item_ids = self.update_item_vectors(item_loader)
        eval_outputs = []
        eval_total_bs = 0
        for eval_step, eval_batch in enumerate(eval_loader):
            eval_batch_size = eval_batch[list(eval_batch.keys())[0]].shape[0]
            metrics = self._eval_batch(eval_batch, item_vectors=self.item_vectors, *args, **kwargs)
            eval_outputs.append((metrics, eval_batch_size))
            eval_total_bs += eval_batch_size
        metrics = self.eval_epoch_end(eval_outputs)
        self._total_eval_samples = eval_total_bs

        return metrics

    
    @torch.no_grad()
    def update_item_vectors(self, item_loader):
        self.model.eval()
        all_item_vectors, all_item_ids = [], []
        model = self.accelerator.unwrap_model(self.model)
        for item_batch in item_loader:
            item_vector = model.item_encoder(item_batch)
            all_item_vectors.append(item_vector)
            all_item_ids.append(item_batch[model.fiid])
        all_item_vectors = self.accelerator.gather_for_metrics(all_item_vectors)
        all_item_ids = self.accelerator.gather_for_metrics(all_item_ids)
        all_item_vectors = torch.cat(all_item_vectors, dim=0)
        all_item_ids = torch.cat(all_item_ids, dim=0).cpu()
        return all_item_vectors, all_item_ids


    def _check_if_eval(self, epoch, step):
        if self.config.evaluation_strategy == 'epoch':
            if (epoch % self.config.eval_interval == 0) and (self._last_eval_epoch != epoch):
                self._last_eval_epoch = epoch
                return True
            return False
        elif self.config.evaluation_strategy == 'step':
            if step % self.config.eval_interval == 0:
                return True
            return False
        else:
            raise ValueError(f'Unknown evaluation strategy: {self.config.evaluation_strategy}')
    

    def _train_batch(self, batch, *args, **kwargs):
        loss_dict = self.model(batch=batch, cal_loss=True, *args, **kwargs)
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
            model = self.accelerator.unwrap_model(self.model)
            outputs = model.eval_step(batch, k=k, *args, **kwargs)
            outputs = self.accelerator.gather_for_metrics(outputs)
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
        if self.model_type == "retriever":
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = ((metric * bs).sum() / bs.sum()).item()
            return out
        else:
            # ranker: AUC, Logloss
            out = {}
            output, bs = zip(*outputs)
            pred, target = zip(*output)
            pred = torch.cat(pred, dim=-1)
            target = torch.cat(target, dim=-1)
            metrics: list = get_eval_metrics(self.config.metrics, self.model_type)
            for metric, func in metrics:
                out[metric] = func(pred, target)
                out[metric] = out[metric].item() if isinstance(out[metric], torch.Tensor) else out[metric]
            return out
            

    @torch.no_grad()
    def compute_metrics(self, output: Tuple):
        """ Compute the metrics given the output of the model.

        Args:
            output (Tuple): The output of the model.

        Returns:
            Dict: The computed metrics.
        """
        metrics: list = get_eval_metrics(self.config.metrics, self.model_type)
        cutoffs = self.config.cutoffs
        output_dict = {}
        if self.model_type == "retriever":
            for metric, func in metrics:
                for cutoff in cutoffs:
                    output_dict[f"{metric}@{cutoff}"] = func(*output, cutoff)
        else:
            output_dict = (output[0].cpu(), output[1].cpu())    # (pred, target)
        return output_dict


    def log_dict(self, d: Dict):
        """Log a dictionary of values."""
        output_list = [f"{k}={v}" for k, v in d.items()]
        self.log_info(", ".join(output_list))

    def log_info(self, *arg, **kwargs):
        if self.accelerator.is_main_process:
            self.logger.info(*arg, **kwargs)


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


    def save_state(self, checkpoint_dir: str):
        """Save the state of the trainer.
        
        Args:
            checkpoint_dir(str): the directory where the checkpoint should be saved.
        """
        # save the parameters of the model
        # save model configuration, enabling loading the model later
        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save(checkpoint_dir)

            if self.model_type == "retriever":
                item_vectors_path = os.path.join(checkpoint_dir, 'item_vectors.pt')
                torch.save({'item_vectors': self.item_vectors, 'item_ids': self.item_ids}, item_vectors_path)

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
            self.log_info(f"Saved the model and trainer state to {checkpoint_dir}.")


    @property
    def state(self):
        state_dict = {
            "global_step": self.global_step,
        }
        return state_dict


    def _check_checkpoint_dir(self):
        if not self.train_mode:
            return
        checkpoint_dir = self.config.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("Checkpoint directory must be specified.")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            # check if the checkpoint_dir is empty
            if len(os.listdir(checkpoint_dir)) > 0:
                raise ValueError(f"Checkpoint directory `{checkpoint_dir}` is not empty.")

    def gradient_clipping(self, clip_norm):
        if (clip_norm is not None) and (clip_norm > 0):
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)


    def parameter_init(self, method, **kwargs):
        if method == "xavier_uniform":
            torch.init.xavier_uniform_(self.model.parameters(), **kwargs)
        elif method == "xavier_normal":
            torch.init.xavier_normal_(self.model.parameters(), **kwargs)
        elif method == "orthogonal":
            torch.init.orthogonal_(self.model.parameters(), **kwargs)
        elif method == "constant":
            torch.init.constant_(self.model.parameters(), **kwargs)
        elif method == "zeros":
            torch.init.zeros_(self.model.parameters(), **kwargs)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
