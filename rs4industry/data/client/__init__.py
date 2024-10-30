from .base import BaseClient
from .hdfs import HDFSClient

CLIENT_MAP = {
    'file': BaseClient,
    'hdfs': HDFSClient
}

def get_client(client_type: str, url: str):
    if client_type in CLIENT_MAP.keys():
        return CLIENT_MAP[client_type](url=url)
    else:
        raise ValueError(f"Unknown client type: {client_type}")

__all__ = [
    "get_client",
    "BaseClient",
    "HDFSClient"
]