#!/usr/bin/env python3.5

"""Train and export a simple Softmax Regression TensorFlow model.
The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    data_dir export_dir
"""

import os
import sys
import shutil

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import builder as saved_model_builder

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1,
                            'version number of the model.')
tf.app.flags.DEFINE_string("data_dir", "/tmp", "The MNIST Data.")
tf.app.flags.DEFINE_string('export_dir', '/tmp', 'Working directory.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 3 or sys.argv[-1].startswith('--model_version') or sys.argv[-1].startswith("--training_iteration"):
        print('Usage: mnist_softmax_export.py [--training_iteration=x] '
              '[--model_version=y] --data_dir=z --export_dir=w')
        sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print("Please specify a positive value for traininig iteration.")
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print("Please specify a positive value for version number.")
        sys.exit(-1)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Train model
    print("Training model...")

    sess = tf.InteractiveSession()

    # Create the model
    # x = tf.placeholder(tf.float32, [None, 784])
    serialized_tf_example = tf.placeholder(tf.string, name="tf_example")
    feature_configs = {"x": tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example["x"], name="x")  # use tf.identity() to assign name
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    tf.global_variables_initializer().run()

    y = tf.nn.softmax(tf.matmul(x, w) + b, name="y")

    # The cross-entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) #-tf.reduce_sum(y_ * tf.log(y))

    # Define loss and optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    values, indices = tf.nn.top_k(y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(10)])
    )

    prediction_classes = table.lookup(tf.to_int64(indices))

    # Train
    for _ in range(FLAGS.training_iteration):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("training accuracy {0}".format(sess.run(
        accuracy, feed_dict={x: mnist.test.images,
                             y_: mnist.test.labels})))
    print('Done training!')

    # Export model
    export_path_base = FLAGS.export_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version))
    )
    print("Exporting trained model to", export_path)

    # remove if already exists
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_signature = tf.saved_model.signature_def_utils.classification_signature_def(
        examples=serialized_tf_example,
        classes=prediction_classes,
        scores=values
    )

    predict_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={"images": x},
        outputs={"scores": y},
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    builder.add_meta_graph_and_variables(
        sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict_images":
                predict_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature
        },
        legacy_init_op=legacy_init_op
    )

    builder.save()

    print("Done exporting!")


if __name__ == '__main__':
    tf.app.run()