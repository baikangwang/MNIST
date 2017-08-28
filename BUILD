# mnist bazel build file

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

py_binary(
    name = "mnist_predict",
    srcs = [
        "mnist_predict.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_serving/example:mnist_input_data",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)