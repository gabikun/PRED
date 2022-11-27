import molReader
import gcn

molecule, featureMat, adjMat, reguMat = molReader.reader('C1CC1C2CC2')
print("reguMat")
print(reguMat)
print("adjMat")
print(adjMat)
print("featureMat")
print(featureMat)

model = gcn.GCN(19)
print(model)
