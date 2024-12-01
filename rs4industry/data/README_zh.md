# 数据集配置说明

### TODO
- [] 需不需要支持对特征的简单处理，如数据中timestamp特征离散化处理？

本文档对数据集配置文件的各个字段设置进行了详细说明，给出通用的配置文件json模板如下：

```json
{
    "name": "数据集名称 (必选)",
    "type": "数据集类型，如 hdfs 或者 file (必选)",
    "url": "数据集交互数据的位置，如hdfs://127.0.0.1:8020/recstudio/recflow/daily_logs (必选)",
    "file_format": "数据存储格式，如 feather, parquet, pkl等。默认为 auto, 即自动检测。",
    "date_format": "时间格式，如 %Y-%m-%d。默认为 %Y-%m-%d。",
    "item_col": "物品id列名 (必选)",
    "context_features": ["使用的上下文特征列表", "特征1", "特征2", "(必选)"],
    "item_features": ["使用的物品特征列表", "context_features 和 item_features 不能重复, 均为主表的列名", "(必选)"],
    "labels": ["使用的标签列表", "多个标签一般表示多任务训练", "标签列表不能为空", "(必选)"],
    "filter_settings": {
        "过滤特征名称": ["过滤条件1", "过滤条件2", "过滤条件形式为 (==, !=, >=, <=, >, <)[number]"],
        "like": ["==1"],
        "用途"： "一般用于按照标签过滤，比如召回模型需要只需要保留label=1的样本，负样本来自于从候选物品集采样"
    },
    "item_info": {
        "url": "召回模型候选物品信息数据存储位置，如 hdfs://127.0.0.1:8020/recstudio/recflow/others/video_info.pkl, 召回模型必选",
        "key": "item id 的列名。对于dataframe形式的文件必须提供, 对于dict形式的文件不需要提供",
        "columns": ["物品特征表的列名", "在item_info中必选", "尤其用于针对dict文件的特征命名"],
        "use_cols": ["item_info中需要使用的特征列表", "如果为空，则使用所有列"]
    }
    "user_sequential_info": {
        "url": "用户序列数据存储位置，如 hdfs://127.0.0.1:8020/recstudio/recflow/daily_logs。设置 user_sequential_info 为 null 则表示不使用独立的序列文件",
        "file_format": "序列数据存储格式，如 feather, parquet, pkl 等。默认为 auto, 即自动检测。",
        "key": "查询序列数据索引的键值，如 request_id。该值还需要在交互数据表中存在。",
        "columns": ["序列特征表的列名", "在item_info中必选", "尤其用于针对dict文件的特征命名", "一般与item_features相同或者为其子集"],
        "use_cols": ["user_sequential_info 中需要使用的特征列表", "如果为空，则使用所有列"]
    },
    "stats": {
        "特征1": 6,
        "特征2": 10,
        "(必选)": "离散特征的数量"
    },
    "train_settings": {
        "start_date": "2024-01-13 (必选)",
        "end_date": "2024-02-08 , 不包含该日期的数据(必选)"
    },
    "test_settings": {
        "start_date": "2024-02-08 (必选)",
        "end_date": "2024-02-09，不包含该日期的数据 (必选)"
    },
    "post_process": {
        "": "暂未实现",
        "特征名称": {
            "operator": "特征的处理函数, 目前支持常见的离散化处理函数。如时间离散化。对于更复杂的离散化处理，建议在特征工程中进行处理后再使用。",
            "parameters": {
                "参数名称": "参数值",
            }
        },
        "request_timestamp": {
            "operator": "time_decompose",
            "parameters": {
                "informat": "timestamp",
                "outformat": ""
            }
        },
        "duration": {
            "operator": "time_decompose",
            "parameters": {
                "informat": "ms",
                "outformat": ""
            }
        }
    }
}
```
