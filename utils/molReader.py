import PRED.utils.utils as utils
import numpy as np
import scipy.linalg
from rdkit import Chem

# features : https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
features = [
    "GetSymbol",
    "GetDegree",
    "GetImplicitValence",
    "GetIsAromatic"
]
atomSymbol = [
    'C', 'O', 'N', 'S', 'Cl', 'P', 'I', 'Na'
]
atomDegree = [
    0, 1, 2, 3, 4
]
atomImplicitValence = [
    0, 1, 2, 3, 4
]
aromatic = [
    True, False
]


def reader(molSmile):
    mol = Chem.MolFromSmiles(molSmile)
    utils.exception("mol is considered as 'None'", mol is None)

    adjacencyMatrix = createAdjacencyMatrix(mol)
    # print(adjacencyMatrix)
    featureMatrix = createFeatureMatrix(mol)
    # print(featureMatrix)
    regularizationMatrix = createRegularizationMatrix(mol)
    # print(regularizationMatrix)
    binaryFeatureMatrix = convertFeatureMatrixToBinary(featureMatrix)
    # print(binaryFeatureMatrix)

    return mol, binaryFeatureMatrix, adjacencyMatrix, regularizationMatrix


def createAdjacencyMatrix(mol):
    return np.array(Chem.GetAdjacencyMatrix(mol))


def createRegularizationMatrix(mol):
    adj_dash = createAdjacencyMatrix(mol) + np.identity(mol.GetNumAtoms())
    deg = np.matrix(np.diag(np.array(np.sum(adj_dash, axis=0))))
    half_deg = scipy.linalg.fractional_matrix_power(deg, -0.5)
    return half_deg.dot(adj_dash).dot(half_deg)


def createFeatureMatrix(mol):
    result = []
    for atom in mol.GetAtoms():
        atomFeatures = []
        for feature in features:
            atomFeatures.append(getattr(atom, feature)())
        result.append(atomFeatures)

    return result


def convertFeatureMatrixToBinary(featureMatrix):
    result = []
    for feature in featureMatrix:
        cursor = 0
        binaryFeature = [0] * 19

        binaryFeature[atomSymbol.index(feature[0]) + cursor] = 1
        cursor += len(atomSymbol)

        binaryFeature[atomDegree.index(feature[1]) + cursor] = 1
        cursor += len(atomDegree)

        binaryFeature[atomImplicitValence.index(feature[2]) + cursor] = 1
        cursor += len(atomImplicitValence)

        if feature[3]:
            binaryFeature[cursor] = 1

        result.append(binaryFeature)

    return result
