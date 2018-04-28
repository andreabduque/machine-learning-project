import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import functions as f

data = pd.read_csv("segmentation.csv", sep=",")

class Partition:
	def __init__(self, nclusters, view):
		self.c = nclusters
		self.view = view

		self.initialize_parameters()
		self.initialize_partitions()

		self.update_prototypes()

	def initialize_parameters(self):
		#Number of variables
		p = len(self.view.columns)
		#A suitable parameter
		self.gamma = (1/f.sigma_squared(self.view.as_matrix()))**p
		#Global weights for each variable
		self.weights = p*[None]
		#Initialize weights
		for i in range(0, p):
			self.weights[i] = self.gamma**(1/p)

		#Set initial random prototypes
		random_sample = self.view.sample(frac=1)[0:7]
		self.prototypes = random_sample.as_matrix()
		self.initial_prototypes = random_sample.index.values

		print("prots iniciais")
		print(self.prototypes)


	def initialize_partitions(self):
		self.elements = {}

		el_not_prot = list(set(range(0, len(view))) - set(self.initial_prototypes))

		for el in el_not_prot:
			dist = float("inf")
			nearest_cluster = 0

			for h in range(0, self.c):
				#Kernel between element and cluster prototype
				x = 2*(1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[el]),
					np.array(self.prototypes[h])))

				if (x < dist):
					nearest_cluster = h
					dist = x

			self.elements[el] = nearest_cluster

	def update_prototypes(self):
		clusters = [[] for _ in range(self.c)]
		for key in self.elements.keys():
			clusters[self.elements[key]].append(key)

		for i in range(0, self.c):
			#Do not fool yourself: Numerator is a vector
			num  =  0
			denom = 0
			for j in range(0, len(clusters[i])):
				denom += f.gaussian_kernel(self.weights, self.view.iloc[clusters[i][j]], self.prototypes[i])
				num   += np.array(denom) * self.view.iloc[clusters[i][j]]

			self.prototypes[i] = num / denom

	def print_prot(self):
		print(self.prototypes)

	def print_weights(self):
		print(self.weights)

	def print_elements(self):
		print(self.elements)


view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]

part = Partition(7, view)
print("prototypes atualizados")
part.print_prot()
print("pesos")
part.print_weights()
print("elementos")
part.print_elements()
