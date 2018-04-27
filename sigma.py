import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def n_dois_a_dois(x):
	return int((x*(x-1))/2)

data = pd.read_csv("segmentation.csv", sep=",")
view = data.drop(axis=1, columns = ["CLASS"])

M = view.as_matrix()
dist_each_pair_vectors = euclidean_distances(M, M)

vetor = (n_dois_a_dois(len(view)))*[None]
it = 0
for i in range(0, len(dist_each_pair_vectors)-1):
	for j in range(i+1, len(dist_each_pair_vectors)):		
		vetor[it] = dist_each_pair_vectors[i][j]
		it = it + 1

primeiro_quantil = int(0.1*n_dois_a_dois(len(view)))
segundo_quantil  = int(0.9*n_dois_a_dois(len(view)))
vetor_ordenado = np.sort(vetor)

sigma = (vetor_ordenado[primeiro_quantil] + vetor_ordenado[segundo_quantil])/2