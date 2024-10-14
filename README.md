# RecStudio-Industry

## 开发目标
### 训练
#### 优先级-1：数据读取(刘奇，黄旭)
- [ ] 数据存储形式：以parquet(https://parquet.apache.org/)，parquet是一种列式存储的数据格式，相比于CSV格式，能够节省存储资源，加快数据读取速度。
- [ ] 存储文件系统：HDFS，先做单机，再做多机分布式
- [ ] 数据读取：改造pytorch原生的数据读取接口，先单机，后多机分布式。参考实现：https://github.com/DeepRec-AI/DeepRec，https://github.com/tensorflow/tensorflow，
	- [ ] 准备数据集，转成parquet格式，配置Hadoop HDFS环境（刘奇）
	- [ ] 数据读取：pyspark(选项-1)（刘奇）
	- [ ] webdatset(选项-2： https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/, https://github.com/webdataset/webdataset）（黄旭）

#### 优先级-2：Embedding优化
- [ ] 动态Embedding Table，动态创建/释放Embedding向量
- [ ] 特征准入/淘汰

### 推理
#### 优先级-1：数据读取(陈晓龙，黎武超）
- [ ] 数据存储形式：以protocol buffer(https://protobuf.dev/)，protocol buffer是一种序列化数据结构的协议。数据压缩率高，适用于数据传输和通信场景。
- [ ] 存储文件系统：Redis(KV数据库)，先做单机，再做分布式。推理时需要根据user_id, item_id取其对应的特征，KV数据库查询速度快，适用于latency要求严格的推理场景。
- [ ] 推理服务：改造pytorch/python的数据读取结构，使其能够从redis中读取相应的特征数据，进而进行推理
- [ ] 召回服务：faiss(https://github.com/facebookresearch/faiss)实现u2i索引召回，i2i召回
	- [ ] 配置redis环境（陈晓龙）
	- [ ] 读取redis中的protobuffer数据到pytorch中进行推理（陈晓龙）
	- [ ] 学会使用protocol buffer，将推理数据集转成protocol buffer格式（黎武超）
	- [ ] 召回阶段的ANN检索，u2i, i2i faiss（黎武超）
#### 优先级-2：推理加速（黎武超)
- [ ] TensorRT：使用TensorRT进行推理加速

### 核心算法测试
- [ ] 召回：sasrec
- [ ] 粗排：dssm
- [ ] 精排：DIN，DCN v2
- [ ] 数据集效果测试，对齐
