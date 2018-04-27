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
		gamma = 1
		
		#Number of variables
		p = len(self.view.columns)
		#Global weights for each variable
		self.weights = p*[None]
		#Initialize weights
		for i in range(0, p):
			self.weights[i] = gamma**(1/p)

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

	return np.exp((-1/2)*argument)

# print(gaussian_kernel(global_weights, np.array(view.iloc[0]), np.array(view.iloc[1])))


view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]

part = Partition(7, view)
part.print_prot()
part.print_weights()