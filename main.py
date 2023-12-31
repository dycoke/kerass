from kerass import *
from sklearn.preprocessing import LabelEncoder, normalize
import pandas as pd

def get_data(path):
    data = pd.read_csv(path)
    cols = list(data.columns)
    target = cols.pop()
    X = data[cols].copy()
    y = data[target].copy()
    y = LabelEncoder().fit_transform(y)
    return np.array(X), np.array(y)

X, y = get_data("datasets/iris.csv")

model = Network()
model.add(DenseLayer(10))
model.add(DenseLayer(10))
model.add(DenseLayer(10))
model.add(DenseLayer(3))
model.train(X, y, 300)