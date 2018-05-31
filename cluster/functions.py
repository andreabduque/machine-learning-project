import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#Receives a vector of wweights and two vectors of different instances
def gaussian_kernel(global_weights, x, y):
	argument = np.dot( (np.array(x) - np.array(y))*(np.array(x) - np.array(y)), global_weights)

	return np.exp((-1/2)*argument)

#Receives feature data in a numpy matrix format
def sigma_squared(feature_matrix):
	distance_matrix = euclidean_distances(feature_matrix, feature_matrix)

	vector_length = int((len(distance_matrix)*(len(distance_matrix)-1))/2)
	vector = vector_length*[None]

	it = 0
	for i in range(0, len(distance_matrix)-1):
		for j in range(i+1, len(distance_matrix)):
			vector[it] = distance_matrix[i][j]**2
			it = it + 1

	first_quantile = int(0.1*vector_length)
	second_quantile  = int(0.9*vector_length)
	sorted_vector = np.sort(vector)

	return 0.5*(sorted_vector[first_quantile] + sorted_vector[second_quantile])
