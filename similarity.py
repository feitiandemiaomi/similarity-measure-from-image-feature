import pandas as pd
import numpy as np

df_features = pd.read_csv("/Users/deveshbatra/Desktop/DPhil/Gallery Game/Pipeline_26_50/docs/combined-features.csv", header=None)
print(len(df_features))
print(len(df_features.iloc[0]))
from sklearn.metrics.pairwise import cosine_similarity

similarity = np.empty([len(df_features),len(df_features)])

for i in range(len(df_features)):
	print(i)
	for j in range(len(df_features)):
		similarity[i][j] = cosine_similarity(df_features.iloc[i,:].reshape(1, -1), df_features.iloc[j,:].reshape(1, -1))

np.savetxt("/Users/deveshbatra/Desktop/DPhil/Gallery Game/Pipeline_26_50/docs/similarity-matrix.csv", similarity)