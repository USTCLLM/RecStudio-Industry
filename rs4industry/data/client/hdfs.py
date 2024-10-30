import os
import fsspec

from rs4industry.data.client.base import BaseClient

class HDFSClient(BaseClient):
    def __init__(self, url: str):
        self._check_hdfs_connection(url)
        super(HDFSClient, self).__init__(url)
        
    @staticmethod
    def _check_hdfs_connection(hdfs_url: str) -> bool:
        """Check that we can connect to the HDFS filesystem at url"""
        if not isinstance(hdfs_url, str):
            raise TypeError(f"Expected `url` to be a string, got {type(hdfs_url)}")
        if not hdfs_url.startswith("hdfs://"):
            raise ValueError(f"Expected `url` to start with 'hdfs://', got {hdfs_url}")
        try:
            fs = fsspec.filesystem('hdfs', fs_kwargs={'hdfs_connect': hdfs_url})
        except ImportError:
            raise ImportError("`fsspec` is not installed")
        except Exception as e:
            print(e)
            raise ValueError(f"Could not connect to {hdfs_url}")
        return True
    
    def list_dir(self) -> list[str]:
        """List all files and directories in the given directory."""
        fs = fsspec.filesystem('hdfs')
        return [os.path.basename(file) for file in fs.ls(self.url)]
    


# test
if __name__ == "__main__":
    url = "hdfs://node1:8020/recstudio/recflow/realshow"
    client = HDFSClient(url)
    print(client.list_dir())
    df = client.load_file("2024-01-14.feather")
    print(df.columns)
    print(df.head(3))

    seq_url = "hdfs://127.0.0.1:8020/recstudio/recflow/seq_effective_50"
    seq_client = HDFSClient(seq_url)
    seq_df = seq_client.load_file("2024-01-14.pkl")
    print(seq_df.columns)
    print(seq_df.head(3))