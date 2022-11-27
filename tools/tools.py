import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
    
def get_data(path):
    data = pd.read_csv(path, index_col=0)

    cols = list(data.columns)
    target = cols.pop()

    X = data[cols].copy()
    y = data[target].copy()

    y = LabelEncoder().fit_transform(y)

    return np.array(X), np.array(y)