# jihun_augment

uploaded python module - 2020/0508


python 3.7에서만 test




pip install opencv-python

pip install PyMySQL

pip install grpcio

pip install grpcio-tools

pip install protobuf 

python3 -m grpc_tools.protoc -I ./proto --python_out=. --grpc_python_out=. ./proto/aug_protobuf.proto
