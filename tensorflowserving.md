# Tensorflow Serving

## Launch serving server

```bash
cd Projects/serving
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
 --port=9000 --model_name=mnist \
 --model_base_path=/home/bkwang/Projects/MNIST/models/
``` 

## Build serving client

```bash
cd Projects/serving
ln -s ~/Projects/MNIST tf_models 
bazel build tensorflow_serving/MNIST/mnist_predict.py
```

## Test the server

```bash
python tensorflow_serving/MNIST/mnist_predict.py \
 --num_tests=1000 --server=localhost:9000 \
 --data_dir=/home/bkwang/Projects/MNIST/input_data
```

