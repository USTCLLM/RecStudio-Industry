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
    "user_sequential_info": {
        "url": "用户序列数据存储位置，如 hdfs://127.0.0.1:8020/recstudio/recflow/daily_logs。设置 user_sequential_info 为 null 则表示不使用独立的序列文件",
        "file_format": "序列数据存储格式，如 feather, parquet, pkl 等。默认为 auto, 即自动检测。",
        "key": "查询序列数据索引的键值，如 request_id。该值还需要在交互数据表中存在。",
        "value": "序列数据值的键值，如 seq。如果数据内容为dict而非DataFrame，即没用列名，该值将作为seq的在batch数据内的名称。"
    },

    "features": ["使用的特征(作为x)列表", "设置为null表示使用除标签特征外的所有特征"],
    "labels": ["使用的标签列表", "多个标签一般表示多任务训练", "标签列表不能为空", "(必选)"],

    "post_process": {
        "request_timestamp": "lambda x: (x - 1578036400) / 24 / 3600",
        "duration": ""
    }

    // TODO:
    // "feature_settings": {
    //     "request_timestamp": "lambda x: ",
    //     "feat_xxx": "特征处理函数python代码，输入是特征值，输出是处理好的特征字典，单个特征可以被处理成多个"
    // }

    "stats": {
        "(必选)"
    },

    "train_settings": {
        "start_date": "2024-01-13 (必选)",
        "end_date": "2024-02-08 (必选)"
    },

    "test_settings": {
        "start_date": "2024-02-08 (必选)",
        "end_date": "2024-02-09 (必选)"
    }
}
```
