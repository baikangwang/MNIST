# Training and Prediction

## Build serving Server

```bash
cd ~/Projects/serving  && \
bazel build -c opt //tensorflow_serving/...
```

## Build serving client

```bash
cd ~/Projects/serving && \
ln -s ~/Projects/MNIST tensorflow_serving && \
bazel build -c opt //tensorflow_serving/MNIST:mnist_predict
```

## Training

```bash
cd ~/Projects/MNIST && \
python mnist_softmax_export.py \
--data_dir=/home/bkwang/Projects/MNIST/input_data \
--export_dir=/home/bkwang/Projects/MNIST/models
```

## Launch serving server

```bash
cd ~/Projects/serving && \
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
 --port=9000 --model_name=mnist \
 --model_base_path=/home/bkwang/Projects/MNIST/models/
``` 

## Test the server

```bash
cd Projects/serving && \
bazel-bin/tensorflow_serving/MNIST/mnist_predict \
 --num_tests=1000 --server=localhost:9000 \
 --data_dir=/home/bkwang/Projects/MNIST/input_data
```

