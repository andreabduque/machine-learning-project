import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#Receives a vector of wweights and two vectors of different instances
def gaussian_kernel(global_weights, x, y):

	argument = 0
	for p_j, weight in enumerate(global_weights):
		argument += weight*(x[p_j] - y[p_j])**2

	return np.exp((-1/2)*argument)

#Receives feature data in a numpy matrix format
def gamma(feature_matrix): # essa funcao calcula sigma, e nao gamma. certo? trocar o nome
	distance_matrix = euclidean_distances(feature_matrix, feature_matrix)

	vector_length = int((len(distance_matrix)*(len(distance_matrix)-1))/2)
	vector = vector_length*[None]

	it = 0
	for i in range(0, len(distance_matrix)-1):
		for j in range(i+1, len(distance_matrix)):		
			vector[it] = distance_matrix[i][j]
			it = it + 1

	first_quantile = int(0.1*vector_length)
	second_quantile  = int(0.9*vector_length)
	sorted_vector = np.sort(vector)

	gamma = (sorted_vector[first_quantile] + sorted_vector[second_quantile])/2

	return gamma

# atualização dos prototipos
def update_g(view, elements, prototypes, global_weights):
	qtd_cluster = len(prototypes)
	cluster = [[] for _ in range(qtd_cluster)]
	# alocando as instancias em seus respectivos clusters, fazendo com que o segundo for não fique n^2.
	for key in elements.keys():
		cluster[elements[key]].append(key) #observar se os cluster tao comecando de 0  ou 1. se comecar de 1 subtrair 1 no acesso

	for i in range(0, qtd_cluster):
		numerador   = 0
		denominador = 0
		for j in range(0, len(cluster[i])):
			denominador += gaussian_kernel(global_weights, view.iloc[cluster[i][j]], prototypes[i])
			numerador   += np.array(denominador) * view.iloc[cluster[i][j]]

		prototypes[i] = numerador / denominador

	return prototypes

# atualizacao dos hyper-parameters
def update_hyper_parameters(view, elements, prototypes, global_weights, gamma):
	qtd_cluster = len(prototypes)
	cluster = [[] for _ in range(qtd_cluster)]

	for key in elements.keys():
		cluster[elements[key]].append(key)

	hyper_parameters = qtd_cluster * [None]

	for j in range(0, qtd_cluster):
		produtorio  = 1
		denominador = 0
		flag = 1 # sem essa flag o denominador seria calculado p vezes

		for h in range(0, qtd_cluster):
			produtorio *= numerador
			numerador = 0

			for i in range(0, qtd_cluster):
				for k in range(0, len(cluster[i])):
					parcela_gaussiana = gaussian_kernel(global_weights, view.iloc[cluster[i][j]], prototypes[i])
					segunda_parcela_numerador = (view.iloc[k][h] - prototypes[i][h]) ** 2

					numerador += parcela_gaussiana * segunda_parcela_numerador

					if(flag == 1):
						segunda_parcela_denominador = (view.iloc[k][j] - prototypes[i][j]) ** 2
						denominador += parcela_gaussiana * segunda_parcela_denominador
			flag = 0

		produtorio = (gamma**(1/qtd_cluster)) * (produtorio ** (1/qtd_cluster))
		hyper_parameters[j] = produtorio / denominador
	
	return hyper_parameters