import os
import pandas as pd


def detect_file_type(path: str) -> str:
    """Detect the file type from its extension"""
    file_extension = path.split('.')[-1]
    if file_extension == "parquet":
        return "parquet"
    elif file_extension == "feather":
        return "feather"
    elif file_extension in ["csv", "txt"]:
        return "csv"
    elif file_extension in ["pkl", "pickle"]:
        return "pkl"
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


class BaseClient(object):
    def __init__(self, url: str):
        self.url = url

    def load_file(self, path=None, **kwargs):
        if not path:
            path = self.url
        else:
            path = os.path.join(self.url, path)
        filetype = detect_file_type(path)
        if filetype == "parquet":
            df = pd.read_parquet(path, **kwargs)
        elif filetype == "csv":
            df = pd.read_csv(path, **kwargs)
        elif filetype == "feather":
            df = pd.read_feather(path, **kwargs)
        elif filetype == "pkl":
            df = pd.read_pickle(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")
        return df
    
    def list_dir(self) -> list[str]:
        """List all files and directories in the given directory."""
        return os.listdir(self.url)
    
# test
if __name__ == "__main__":
    url = "hdfs://127.0.0.1:8020"
    path = "/recstudio/recflow/daily_logs"
    loader = BaseClient(url)
    print(loader.list_dir())
    df = loader.load_hdfs_file(f"{url}/{path}/2024-01-14.feather")
    print(df.head(3))