import molReader
import gcn
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

molecule, featureMat, adjMat, reguMat = molReader.reader('C1CC1C2CC2')
print("reguMat")
print(reguMat)
print("adjMat")
print(adjMat)
print("featureMat")
print(featureMat)


# Initialize the weights
np.random.seed(77777)
n1 = 15  #number of neurons in the hidden layer
n2 = 20  #number of neurons in the output layer
n3 = 27  #number of neurons in the output layer
n4 = 36  #number of neurons in the output layer

W1 = np.random.randn(19, n1) * 0.01
W2 = np.random.randn(n1, n2) * 0.01
W3 = np.random.randn(n2, n3) * 0.01
W4 = np.random.randn(n3, n4) * 0.01

# Build GCN layer
# In this function, we implement numpy to simplify

def gcn(featMat, W):
    eq = reguMat.dot(featMat).dot(W)

    eq = F.selu(torch.tensor(eq))
    return eq


# Do forward propagation
H1 = gcn(featureMat, W1)
H2 = gcn(H1, W2)
H3 = gcn(H2, W3)
H4 = gcn(H3, W4)
print('Features Representation from GCN output:\n', H4)

def plot_features(output_tensor):
    output_matrix = output_tensor.numpy()
    # Plot the features representation
    x = output_matrix[:, 0]
    y = output_matrix[:, 1]

    size = 1000

    plt.scatter(x, y, size)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('Feature Representation Dimension 0')
    plt.ylabel('Feature Representation Dimension 1')
    plt.title('Feature Representation')

    for i, row in enumerate(output_matrix):
        str = "{}".format(i)
        plt.annotate(str, (row[0], row[1]), fontsize=18, fontweight='bold')

    plt.show()


plot_features(H4)
