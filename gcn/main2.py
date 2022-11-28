import gcnModel
import PRED.utils.molReader as molReader
import numpy as np
import torch.nn.functional as F


molecule, featureMat, adjMat, reguMat = molReader.reader('C1CC1C2CC2')
print("reguMat")
print(reguMat)
print("adjMat")
print(adjMat)
print("featureMat")
print(featureMat)

gcn_model = gcnModel.GCNNetwork(
    n_inputs=19,
    n_outputs=175,
    n_layers=4,
    hidden_sizes=[15, 20, 27, 36],
    activation=np.tanh,
    seed=100
)
print(gcn_model)

y_pred = gcn_model.forward(reguMat, featureMat)
print(y_pred)

vertexF = y_pred.sum(axis=0)
print(vertexF)
