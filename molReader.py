import utils
import numpy as np
from rdkit import Chem

# TODO completer les features : https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
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
    print(adjacencyMatrix)
    featureMatrix = createFeatureMatrix(mol)
    print(featureMatrix)
    regularizationMatrix = createRegularizationMatrix(mol)
    print(regularizationMatrix)

    # TODO function to convert the values of the featureMatrix to binaries values
    binaryFeatureMatrix = convertFeatureMatrixToBinary(featureMatrix)
    print(binaryFeatureMatrix)

def createAdjacencyMatrix(mol):
    return np.array(Chem.GetAdjacencyMatrix(mol))

def createRegularizationMatrix(mol):
    adj = createAdjacencyMatrix(mol)
    deg = np.array(np.sum(adj, axis=0))
    deg = np.matrix(np.diag(deg))
    adj = adj + np.identity(mol.GetNumAtoms())
    return np.matmul(adj, np.linalg.inv(deg))

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

        if (feature[3]): binaryFeature[cursor] = 1

        result.append(binaryFeature)

    return result




