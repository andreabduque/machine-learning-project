import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv("segmentation.csv", sep=",")

class Partition:
	def __init__(self, nclusters, view):
		self.c = nclusters
		self.view = view

		self.initialize_parameters()

	def initialize_parameters(self):		
		#Number of variables
		p = len(self.view.columns)

		gamma = 1
		# gamma = (1/sigma)**p
		#Global weights for each variable
		self.weights = p*[None]
		#Initialize weights
		for i in range(0, p):
			self.weights[i] = gamma**(1/p)
			# self.weights[i] = 1/sqrt(gamma**(1/p))

		#Set initial random prototypes
		self.prototypes = self.view.sample(frac=1)[0:7].index.values

	def set_partitions(self, weights):
		pass

	def print_prot(self):
		print(self.prototypes)

	def print_weights(self):
		print(self.weights)

def gaussian_kernel(global_weights, x, y):

	argument = 0
	for p_j, weight in enumerate(global_weights):
		argument += weight*(x[p_j] - y[p_j])**2
		# argument += (1/(weight**2))*((x[p_j] - y[p_j])**2)

	return np.exp((-1/2)*argument)

# print(gaussian_kernel(global_weights, np.array(view.iloc[0]), np.array(view.iloc[1])))


view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]

part = Partition(7, view)
part.print_prot()
part.print_weights()

def clustering(view, qtd_cluster, global_weights, prototypes):
	cluster = np.empty([qtd_cluster, len(view)])

	for i in range (0, len(view)):
		menor_cluster = infinito
		object_i = 0
		for j in range (0, qtd_cluster):
			x = 2*(1 - gaussian_kernel(global_weights, view.iloc[i], prototypes[j]))
			if (x < menor_cluster):
				menor_cluster = x
				object_i = j

		cluster[object_i].append = i