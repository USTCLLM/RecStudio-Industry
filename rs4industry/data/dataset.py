from __future__ import absolute_import, division, print_function, unicode_literals

from dataclasses import dataclass
from typing import List, Union
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset
from multiprocessing import Process, Queue, Pool
from accelerate import Accelerator

from rs4industry.data.client import get_client
from rs4industry.data.utils.constants import REQUIRED_DATA_CONFIG, DEFAULT_CONFIG
from rs4industry.data.utils.utils import (extract_timestamp,
                                          nested_dict_update,
                                          df_to_tensor_dict,
                                          read_json,
                                          process_conditions)


class Statistics(object):
    @staticmethod
    def from_dict(d: dict):
        stat = Statistics()
        for k, v in d.items():
            setattr(stat, k.strip(), v)
        return stat

    def add_argument(self, name, value):
        setattr(self, name, value)

@dataclass
class DataAttr4Model:
    """
    Data attributes for a dataset. Serve for models
    """
    fiid: str
    flabels: List[str]
    features: List[str]
    context_features: List[str]
    item_features: List[str]
    seq_features: List[str]
    num_items: int  # number of candidate items instead of maximum id of items
    stats: Statistics

    @staticmethod
    def from_dict(d: dict):
        if "stats" in d:
            d["stats"] = Statistics.from_dict(d["stats"])
        attr = DataAttr4Model(**d)
        return attr

    def to_dict(self):
        d = self.__dict__
        for k, v in d.items():
            if type(v) == Statistics:
                d[k] = v.__dict__
        return d



class ItemDataset(Dataset):
    def __init__(self, item_feat_df: torch.Tensor):
        super().__init__()
        self.item_feat_df = item_feat_df
        self.item_ids = item_feat_df.index.to_numpy()
        self.item_id_2_offset = pd.DataFrame.from_dict(
            {item_id: i for i, item_id in enumerate(self.item_ids)},
            orient="index",
            columns=["offset"]
        )
        self.item_id_col = self.item_feat_df.index.name

    def __getitem__(self, index):
        feat_dict = self.item_feat_df.loc[self.item_ids[index]].to_dict()
        if self.item_id_col is not None:
            feat_dict.update({self.item_id_col: self.item_ids[index]})
        return feat_dict
    
    def __len__(self):
        return len(self.item_ids)
    

    def get_item_feat(self, item_ids: torch.LongTensor):
        # TODO: efficiency issue due to item_ids may be in GPU and the first dimension is batch size
        item_ids_list = item_ids.view(-1).tolist()
        item_ids_list = self.item_ids[item_ids_list]
        feats = self.item_feat_df.loc[item_ids_list].to_dict("list")
        feats = {k: torch.tensor(v).reshape(*item_ids.shape).to(item_ids.device) for k,v in feats.items()}
        return feats


class ConfigProcessor(object):
    """
    This class is used to split the data into train and test sets at file level.

    Args:
        config (Union[dict, str]): Configuration dictionary or path to JSON file
    """
    def __init__(self, config: Union[dict, str]):
        self.config = self._load_config(config)
        self._check_config()
        self.data_dir = self.config['url']
        self.train_start_date, self.train_end_date = self._get_date_range(mode='train')
        self.test_start_date, self.test_end_date = self._get_date_range(mode='test')
        self.train_files = self._get_valid_filenames(self.train_start_date, self.train_end_date)
        self.test_files = self._get_valid_filenames(self.test_start_date, self.test_end_date)
        self.attrs = self.get_attrs()


    def get_attrs(self):
        stats = Statistics.from_dict(self.config['stats'])
        def remove_whitespace(string):
            return string.strip()
        self.config['context_features'] = list(map(remove_whitespace, self.config['context_features']))
        self.config['item_features'] = list(map(remove_whitespace, self.config['item_features']))
        if self.config.get("user_sequential_info", None):
            self.config['seq_features'] = list(map(remove_whitespace, self.config['user_sequential_info']['use_cols']))
        else:
            self.config['seq_features'] = []
        return DataAttr4Model(
            fiid=self.config['item_col'],
            flabels=self.config['labels'],
            features=self.config['context_features']+self.config['item_features'],
            context_features=self.config['context_features'],
            item_features=self.config['item_features'],
            seq_features=self.config['seq_features'],
            num_items=getattr(stats, self.config['item_col']),
            stats=stats,
        )

    def _load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dictionary or a JSON file."""
        if isinstance(config, dict):
            pass
        elif isinstance(config, str):
            config = read_json(config)
        else:
            raise TypeError("Config must be a dictionary or path to JSON file.")
        # update nested config to default config
        config = nested_dict_update(DEFAULT_CONFIG, config)
        return config

    def _check_config(self):
        """Checks that all required keys are present in the configuration."""
        def _flatten_keys(dictionary):
            # flatten nested keys with /
            valid_keys = []
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    valid_keys.extend([f"{key}/{subkey}" for subkey in _flatten_keys(value)])
                else:
                    valid_keys.append(key)
            return valid_keys
        
        valid_keys = _flatten_keys(self.config)
        missing_keys = []
        for key in REQUIRED_DATA_CONFIG:
            if key not in valid_keys:
                missing_keys.append(key)
        assert key in valid_keys, f"Missing required keys: {missing_keys}"

    def _get_date_range(self, mode: str):
        """Get start and end date from config
        Args:
            mode (str): train or test
        """
        date_format: str = self.config['date_format']
        start_date: datetime = datetime.strptime(self.config[f'{mode}_settings']['start_date'], date_format)
        end_date: datetime = datetime.strptime(self.config[f'{mode}_settings']['end_date'], date_format)
        return start_date, end_date

    def _get_valid_filenames(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Get all used filenames among given date range in the dataset"""
        data_client = get_client(self.config['type'], self.config['url'])
        filenames = sorted(data_client.list_dir())
        dates = [extract_timestamp(filename) for filename in filenames]
        file_idx = [idx for idx, date in enumerate(dates) if start_date <= date < end_date]
        valid_filenames = [filenames[idx] for idx in file_idx]
        return valid_filenames
    

class DailyDataIterator(object):
    """This class is used to iterate over each day of data (dataframe)
    
    Args:
        config (dict): config dictionary
        filenames (List[str]): list of filenames to iterate over
    """
    def __init__(self, config, filenames):
        self.config = config
        self.data_client = get_client(config['type'], config['url'])
        if config['user_sequential_info']:
            # use given key column and sequence data file
            self.seq_index_col = config['user_sequential_info']['key']
            self.seq_data_client = get_client(config['type'], config['user_sequential_info']['url'])
        else:
            # use given key column, no additional files needed
            self.seq_index_col = None
            self.seq_data_client = None

        if config['item_info']:
            self.item_col = config['item_info']['key']
            self.item_data_client = get_client(config['type'], config['item_info']['url'])
        else:
            self.item_col = None
            self.item_data_client = None

        self.filenames = filenames
        self.cur_idx = 0
        self._data_cache = {'filename': None, 'data': None, 'seq': None}

    def _columns_rename(self, df: pd.DataFrame):
        """ Rename columns to remove whitespace and modify `features` in the config consistently"""
        rename_map = {}
        for col in df.columns:
            new_col = col.strip()
            rename_map[col] = new_col
        self.config['features'] = self.config['context_features'] + self.config['item_features']
        return df.rename(columns=rename_map)

    def load_file(self, filename: str):
        """Load one day of data"""
        data = self.data_client.load_file(filename)
        data = self._columns_rename(data)
        # use given columns
        if self.config['features']:
            if self.seq_index_col is not None and self.seq_index_col not in self.config['features']:
                # add key column for merge
                keep_cols = [self.seq_index_col] + self.config['features'] + self.config['labels']
            else:
                keep_cols = self.config['features'] + self.config['labels']
            data = data[keep_cols]
        data = self.post_process(data)
        return data
    
    def load_sequence_file(self, filename: str):
        """Load one day of sequence data"""
        if self.seq_data_client is None:
            return None
        date_str = filename.split('.')[0]
        fileformat = self.config['user_sequential_info'].get('file_format', 'auto')
        if fileformat != 'auto':
            filename = f"{date_str}.{fileformat}"
        else:
            # auto mode: find the file with the same date
            all_files = self.seq_data_client.list_dir()
            for file in all_files:
                if date_str in file:
                    filename = file
                    break
        data = self.seq_data_client.load_file(filename)
        # sequence data is can be DataFrame or Dict
        if isinstance(data, pd.DataFrame):
            # data.set_index(self.seq_index_col, inplace=True)
            # data = data.to_dict(orient='index')
            pass
        elif isinstance(data, dict):
            # L*N -> n number of L
            data = {k: [v[:, i] for i in range(v.shape[-1])] for k, v in data.items()}
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['user_sequential_info']['columns'])
        else:
            raise ValueError("Sequence data must be DataFrame or Dict")
        # data.set_index(self.seq_index_col, inplace=True)
        if self.config['user_sequential_info'].get('use_cols', None) is not None:
            data = data[self.config['user_sequential_info']['use_cols']]
        return data

    def load_item_file(self):
        """Load all item data"""
        if self.config['item_info'] is None:
            return None
        data = self.item_data_client.load_file()
        if isinstance(data, pd.DataFrame):
            pass
        elif isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['item_info']['columns'])
            # data.set_index(self.config['item_info']['key'], inplace=True)
        else:
            raise ValueError("Item data must be DataFrame or Dict")
        # data = {k: torch.tensor(list(v), dtype=torch.int64) for k, v in data.items()}
        if self.config['item_info'].get('use_cols', None) is not None:
            data = data[self.config['item_info']['use_cols']]
        # data.set_index(self.config['item_info']['key'], inplace=True)
        return data
        
    
    def post_process(self, df):
        # filtering data in log table
        # case1: for retrieval model training
        if self.config['filter_settings'] is not None:
            for col, conditions in self.config['filter_settings'].items():
                condition_func = process_conditions(conditions)
                df = df.loc[df[col].apply(condition_func)]
        if self.config['post_process'] is None:
            return df
        else:
            raise NotImplementedError("Post process is not implemented yet")

    def load_one_day_data(self):
        filename = self.filenames[self.cur_idx]
        return self.load_file(filename), self.load_sequence_file(filename)

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx < len(self.filenames):
            cache_filename = self._data_cache.get('filename', None)
            if cache_filename is not None and self.filenames[self.cur_idx] == cache_filename:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from cache")
                data, seq_data = self._data_cache['data'], self._data_cache['seq']
            else:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from source")
                data, seq_data = self.load_one_day_data()
                self._data_cache['filename'] = self.filenames[self.cur_idx]
                self._data_cache['data'] = data
                self._data_cache['seq'] = seq_data
            self.cur_idx += 1
            return data, seq_data
        else:
            raise StopIteration
        
    def preload_start(self):
        def _load_data(queue):
            data, seq = self.load_one_day_data()
            queue.put((data, seq))
        print(f"Start preload file {self.filenames[self.cur_idx]}")
        queue = Queue()
        p = Process(target=_load_data, args=(queue,))
        return queue, p

    def preload_end(self, queue, p):
        data, seq = queue.get()
        self._data_cache['filename'] = self.filenames[self.cur_idx]
        self._data_cache['data'] = data
        self._data_cache['seq'] = seq
        p.join()



class DailyDataset(IterableDataset):
    def __init__(self, daily_iterator: DailyDataIterator, attrs, shuffle=False, preload=False, seed=42, **kwargs):
        super(DailyDataset, self).__init__(**kwargs)
        accelerator = Accelerator()
        self.rank = accelerator.process_index
        self.num_processes = accelerator.num_processes
        self.seed = seed
        self.daily_iterator = daily_iterator
        self.config = daily_iterator.config
        self.attrs = attrs
        self.shuffle = shuffle
        self.preload = preload
        self.preload_ratio = 0.8
        self.seq_index_col = daily_iterator.seq_index_col
        item_data = daily_iterator.load_item_file()
        self.item_feat_dataset = ItemDataset(item_data) if item_data is not None else None
        self.attrs.num_items = len(self.item_feat_dataset) if item_data is not None else None


    def __iter__(self):
        self.seed += 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError("Not support `num_workers`>0 now.")
        for log_df, seq_df in self.daily_iterator:
            if self.shuffle:
                log_df = log_df.sample(frac=1).reset_index(drop=True)
            tensor_dict = df_to_tensor_dict(log_df)
            num_samples = log_df.shape[0]
            
            if self.preload:
                # `preload_index` controls when to preload
                preload_index = int(self.preload_ratio * num_samples)
            for i in range(num_samples):
                if self.preload and i == preload_index:
                    # preload the next-day logs
                    queue, p = self.daily_iterator.preload_start()
                    # p = Process(target=self.daily_iterator.preload)
                    # p.start()
                if self.preload and (i == num_samples-1):
                    # wait for the preload process
                    self.daily_iterator.preload_end(queue, p)
                    # p.join()

                data_dict = {k: v[i] for k, v in tensor_dict.items()}
                if seq_df is not None:
                    seq_data_dict = seq_df.loc[data_dict[self.seq_index_col].item()].to_dict()
                    data_dict['seq'] = seq_data_dict
                yield data_dict

    def get_item_feat(self, item_ids: torch.Tensor):
        """
        Return item features by item_ids

        Args:
            item_ids (torch.Tensor): [B x N] or [N]

        Returns:
            torch.Tensor: [B x N x F] or [N x F]
        """
        return self.item_feat_dataset.get_item_feat(item_ids)
    
    def get_item_loader(self, batch_size, num_workers=0, shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def get_datasets(config: Union[dict, str]):
    cp = ConfigProcessor(config)

    train_data_iterator = DailyDataIterator(cp.config, cp.train_files)
    test_data_iterator = DailyDataIterator(cp.config, cp.test_files)

    train_data = DailyDataset(train_data_iterator, shuffle=True, attrs=cp.attrs, preload=False)
    test_data = DailyDataset(test_data_iterator, shuffle=False, attrs=cp.attrs, preload=False)
    return (train_data, test_data), cp.attrs


# test
if __name__ == '__main__':
    config = "./examples/data/recflow.json"
    cp = ConfigProcessor(config)
    print(cp.train_start_date, cp.train_end_date)
    print(cp.test_start_date, cp.test_end_date)
    print(cp.train_files)

    train_data_iterator = DailyDataIterator(cp.config, cp.train_files)
    test_data_iterator = DailyDataIterator(cp.config, cp.test_files)
    print(len(train_data_iterator), len(test_data_iterator))

    train_data = DailyDataset(train_data_iterator, shuffle=False, attrs=cp.attrs, preload=False)
    test_data = DailyDataset(test_data_iterator, shuffle=False, attrs=cp.attrs, preload=False)

    for i, data in enumerate(train_data):
        print({k: v.shape for k, v in data.items()})
        if i > 3:
            break

    # data_iter = iter(train_data)
    # sample = next(data_iter)
    # sample = next(data_iter)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, num_workers=0)

    for i, batch in enumerate(train_loader):
        print({k: v.shape for k, v in batch.items()})
        # if i > 3:
        #     break