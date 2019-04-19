import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
scaler = StandardScaler()
train_X = np.random.rand(1000, 500)
train_X = scaler.fit_transform(train_X)
pca_50 = PCA(n_components=50)
pca_data = pca_50.fit_transform(train_X)
print(pca_data)
