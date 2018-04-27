import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv("segmentation.csv", sep=",")


def get_prototypes(nprototypes, view):
	gamma = 1
	print(view.sample(frac=1)[0:7])


def set_initial_parameters():
	pass

def n_dois_a_dois(x):
	return (x*(x-1))/2



groups_class = data["CLASS"]
#get_prototypes(7, data)

#primeiro elemento
view = data.drop(axis=1, columns = ["CLASS"])


# print (len(view.columns))

# print(dist)
tam = len(view)
tam2 = len(view) - 1
it = 0

M = view.as_matrix()
teta = euclidean_distances(M, M)

k = int(n_dois_a_dois(len(view)))
vetor = (k)*[None]

for i in range(0, len(teta)-1):
	for j in range(i+1, len(teta)):		
		vetor[it] = teta[i][j]
		it = it + 1

# for i in range(0, len(teta)):
# 	for j in range(0, len(teta)):
# 		if(j > i):
# 			vetor[it] = teta[i][j]
# 			it += 1

primeiro_quantil = int(0.1*n_dois_a_dois(len(view)))
segundo_quantil  = int(0.9*n_dois_a_dois(len(view)))
vetor_ordenado = np.sort(vetor)
# print(len(vetor_ordenado))

print(vetor_ordenado[primeiro_quantil])
print(vetor_ordenado[segundo_quantil])

# for i in range(1, tam):
# 	for j in range(1, tam):
# 		if(i != j and i > j)
# 		it = it + 1
# 		k[it] = teta[i][j]


# print (np.shape(teta))
# primeiro_quantil = int(0.1*n_dois_a_dois(len(view)))
# segundo_quantil  = int(0.9*n_dois_a_dois(len(view)))
# # vetor_ordenado = np.sort(vetor)

# print(vetor_ordenado[primeiro_quantil])
# print(vetor_ordenado[segundo_quantil])

# sigma = (vetor_ordenado[primeiro_quantil] + vetor_ordenado[segundo_quantil])/2
# print(sigma)
# print(str(i) + ' ' + str(j)) #vetor.append(np.linalg.norm(i - j))