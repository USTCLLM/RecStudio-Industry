
# create proto
protoc --python_out=. ./inference/feature_insert/protos/recflow.proto
echo "recflow.proto created !!!"

# insert data
python ./inference/feature_insert/recflow_script/insert_redis.py