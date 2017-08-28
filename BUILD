# PubFig bazel build file

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
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
    imports = [
    ],
    data = glob([])
)