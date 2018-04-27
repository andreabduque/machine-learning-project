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


	def initialize_parameters(self):		
		#Number of variables
		p = len(self.view.columns)
		#A suitable parameter
		gamma = f.gamma(self.view.as_matrix())
		#Global weights for each variable
		self.weights = p*[None]
		#Initialize weights
		for i in range(0, p):
			self.weights[i] = gamma**(1/p)

		#Set initial random prototypes
		self.prototypes = self.view.sample(frac=1)[0:7].index.values


	def initialize_partitions(self):
		self.elements = {}

		el_not_prot = list(set(range(0, len(view))) - set(self.prototypes))

		for el in el_not_prot:
			dist = float("inf")
			nearest_cluster = 0
			
			for h in range(0, self.c):
				#Kernel between element and cluster prototype
				x = 2*(1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[h]),
					np.array(self.view.iloc[self.prototypes[h]])))

				if (x < dist):
					nearest_cluster = h
					dist = x
			
			self.elements[el] = nearest_cluster

	def print_prot(self):
		print(self.prototypes)

	def print_weights(self):
		print(self.weights)

	def print_elements(self):
		print(self.elements)


# print(f.gaussian_kernel(global_weights, np.array(view.iloc[0]), np.array(view.iloc[1])))


view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]

part = Partition(7, view)
part.print_prot()
part.print_weights()
part.print_elements()