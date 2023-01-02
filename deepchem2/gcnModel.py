from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import tensorflow.keras.layers as layers

class MyGraphConvModel(tf.keras.Model):

    def __init__(self, n_tasks, batch_size):
        super(MyGraphConvModel, self).__init__()

        # Message Passing Layers
        self.gc1 = GraphConv(15, activation_fn=tf.nn.selu)
        self.gc2 = GraphConv(20, activation_fn=tf.nn.selu)
        self.gc3 = GraphConv(27, activation_fn=tf.nn.selu)
        self.gc4 = GraphConv(36, activation_fn=tf.nn.selu)
        self.gp1 = GraphPool()

        # Readout
        self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
        self.batch_norm3 = layers.BatchNormalization()
        self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

        self.dense2 = layers.Dense(n_tasks * 2)
        self.logits = layers.Reshape((n_tasks, 2))
        self.softmax = layers.Softmax()

    def call(self, inputs):
        # Message Passing Layers
        gc1_output = self.gc1(inputs)
        gc2_output = self.gc2([gc1_output] + inputs[1:])
        gc3_output = self.gc3([gc2_output] + inputs[1:])
        gc4_output = self.gc4([gc3_output] + inputs[1:])
        gp1_output = self.gp1([gc4_output] + inputs[1:])

        # Readout
        dense1_output = self.dense1(gp1_output)
        batch_norm3_output = self.batch_norm3(dense1_output)
        readout_output = self.readout([batch_norm3_output] + inputs[1:])

        logits_output = self.logits(self.dense2(readout_output))
        return self.softmax(logits_output)