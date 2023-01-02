from deepchem.models.layers import GraphConv, GraphPool, GraphGather
import tensorflow as tf
import tensorflow.keras.layers as layers
import torch.nn as nn
import torch.nn.functional as F

class MyGraphConvModel(tf.keras.Model):

    def __init__(self, n_tasks, batch_size):
        super(MyGraphConvModel, self).__init__()

        # Hidden layers
        self.gc1 = GraphConv(15, activation_fn=tf.nn.selu)
        self.gc2 = GraphConv(20, activation_fn=tf.nn.selu)
        self.gc3 = GraphConv(27, activation_fn=tf.nn.selu)
        self.gc4 = GraphConv(36, activation_fn=tf.nn.selu)

        # Readout
        self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.selu)

        self.dense2 = layers.Dense(n_tasks * 2)
        self.logits = layers.Reshape((n_tasks, 2))
        self.softmax = layers.Softmax()

        # Fully connected layers
        self.linear1 = nn.Linear(in_features=175, out_features=96)
        self.bn1 = nn.BatchNorm1d(num_features=96)
        self.linear2 = nn.Linear(in_features=96, out_features=63)
        self.bn2 = nn.BatchNorm1d(num_features=63)
        self.linear3 = nn.Linear(in_features=63, out_features=138)

    def call(self, inputs):
        # TODO inputs
        # Hidden layers
        gc1_output = self.gc1(inputs)
        gc2_output = self.gc2([gc1_output] + inputs[1:])
        gc3_output = self.gc3([gc2_output] + inputs[1:])
        gc4_output = self.gc4([gc3_output] + inputs[1:])

        # TODO readout
        # Readout
        readout_output = self.readout([gc4_output] + inputs[1:])

        logits_output = self.logits(self.dense2(readout_output))

        rd_output = self.softmax(logits_output)


        # Fully connected layers
        fc1_output = F.relu(self.bn1(self.linear1(rd_output)))
        fc2_output = F.relu(self.bn2(self.linear2(fc1_output)))
        return F.sigmoid(self.linear3(fc2_output))
