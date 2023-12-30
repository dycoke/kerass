from kerass import *
from sklearn.preprocessing import LabelEncoder
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
model.add(DenseLayer(6))
model.add(DenseLayer(8))
model.add(DenseLayer(10))
model.add(DenseLayer(3))
model._compile(X)

model._init_weights(X)
print(model.params[0]['W'].shape, model.params[0]['b'].shape)
print(model.params[1]['W'].shape, model.params[1]['b'].shape)
print(model.params[2]['W'].shape, model.params[2]['b'].shape)
print(model.params[3]['W'].shape, model.params[3]['b'].shape)