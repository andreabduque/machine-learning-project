import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import adjusted_rand_score
import functions as f
import json
import time
from multiprocessing import Pool

class Partition:
	def __init__(self, nclusters, view):
		self.c = nclusters
		self.view = view

	def run(self):
		self.initialize_parameters()
		self.initialize_partitions()
		self.optimize_partition()

	def get_objective_function(self):
		clusters = [[] for _ in range(self.c)]
		for key in self.elements.keys():
			clusters[self.elements[key]].append(key)

		energy = 0
		for i in range(self.c):
			k_lim = len(clusters[i])
			for k in range(k_lim):
				energy += (1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[clusters[i][k]]),
					np.array(self.prototypes[i])))

		return energy


	def initialize_parameters(self):
		#Number of variables
		self.p = len(self.view.columns)
		#Number of rows
		self.rows = len(self.view)
		#A suitable parameter
		self.gamma = (1/f.sigma_squared(self.view.as_matrix()))**self.p
		#Global weights for each variable
		self.weights = self.p*[None]
		#Initialize weights
		for i in range(self.p):
			self.weights[i] = self.gamma**(1/self.p)

		#Set initial random prototypes
		random_sample = self.view.sample(frac=1)[0:(self.c)]
		self.prototypes = random_sample.as_matrix()
		self.initial_prototypes = random_sample.index.values

	def initialize_partitions(self):
		self.elements = {}
		self.clusters = [[] for _ in range(self.c)]
		self.item_to_position = clusters = [{} for _ in range(self.c)]

		for el in range(self.rows):
			dist = float("inf")
			nearest_cluster = 0

			for h in range(self.c):
				#Kernel between element and cluster prototype
				x = 2*(1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[el]),
					np.array(self.prototypes[h])))

				if (x < dist):
					nearest_cluster = h
					dist = x

			self.elements[el] = nearest_cluster
			self.clusters[nearest_cluster].append(el)
			self.item_to_position[nearest_cluster][el] = len(self.clusters[nearest_cluster])-1

	#Compute best prototypes and best hyper parameters
	#Returns partition energy
	def optimize_partition(self):
		test = 1
		ite = 0
		while(test):
			ite += 1
			# clusters = [[] for _ in range(self.c)]
			# for key in self.elements.keys():
			# 	clusters[self.elements[key]].append(key)

			#Calculating best prototypes
			self.update_prototypes(self.clusters)
			#calculating hyper parameters
			self.update_weights(self.clusters)

			#Allocating in partitions
			test = 0
			for k in range(self.rows):
				dist = float("inf")
				nearest_cluster = 0
				#Kernel between element and cluster prototype
				for h in range(self.c):
					x = 2*(1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[k]),
						np.array(self.prototypes[h])))
					if (x < dist):
						nearest_cluster = h
						dist = x

				if(nearest_cluster != self.elements[k]):
					test = 1
					nearest_cluster_old = self.elements[k]
					self.elements[k] = nearest_cluster
					#Adding to new cluster
					self.clusters[nearest_cluster].append(k)
					self.item_to_position[nearest_cluster][k] = len(self.clusters[nearest_cluster])-1

					#Removing from old cluster
					position = self.item_to_position[nearest_cluster_old].pop(k)
					last_item = self.clusters[nearest_cluster_old].pop()
					if position != len(self.clusters[nearest_cluster_old]):
						self.clusters[nearest_cluster_old][position] = last_item
						self.item_to_position[nearest_cluster_old][last_item] = position


	def update_prototypes(self, clusters):
		for i in range(self.c):
			#Do not fool yourself: Numerator is a vector
			num  =  0
			denom = 0
			j_lim = len(clusters[i])
			for j in range(j_lim):
				kernel = f.gaussian_kernel(self.weights, self.view.iloc[clusters[i][j]], self.prototypes[i])
				num   += kernel * self.view.iloc[clusters[i][j]]
				denom += kernel

			if(denom != 0):
				self.prototypes[i] = num / denom
			else:
				print("denominador igual a zero no update do prototipo!")

	def update_weights(self, cluster):
		new_weights = self.p*[None]

		vetor_somas = np.zeros(self.p)
		#Calculo denominador
		for h in range(self.p):
			sum_kernels = 0			
			for i in range(self.c):
				k_lim = len(cluster[i])
				for k in range(k_lim):
					#An Element from cluster
					x_k = np.array(self.view.iloc[cluster[i][k]])
					#Prototype from cluster
					g_i =  self.prototypes[i]		

					sum_kernels += f.gaussian_kernel(self.weights, x_k, g_i) * (x_k[h] - g_i[h]) * (x_k[h] - g_i[h])


			vetor_somas[h] = sum_kernels

		numerador = (self.gamma**(1/self.p))*np.prod(vetor_somas**(1/self.p))
		self.weights = np.divide(numerador, vetor_somas)

	def get_result(self):
		return self.elements

	def get_partition_rand_index(self, true_labels):
		pred_labels = self.rows*[None]
		for i, key in enumerate(sorted(self.elements)):
			pred_labels[i] = self.elements[key]

		return adjusted_rand_score(true_labels, pred_labels)

def get_clustering_result(it):
	part = Partition(n_clusters, view)
	part.run()
	print("fim iteracao")
	return (part.get_objective_function(), part.get_partition_rand_index(groups_class), part.initial_prototypes, 
			part.prototypes, part.weights, part.get_result())
	
n_clusters = 3
data = pd.read_csv("../iris.data", sep=",")
view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]

before = time.time()

with Pool(processes=5) as pool:
	results = pool.map(get_clustering_result, 100*[None]) 

# print(time.time() - before)
print("Tempo " + str(time.time() - before))

results.sort(key=lambda tup: tup[0])

print("Melhor Energia " + str(2*results[0][0]))
print("ARI " + str(results[0][1]))
print("Prototipos Iniciais")
print(results[0][2])
print("Representantes de cada grupo")
print(results[0][3])
print("Hiperparametros")
print(results[0][4])

with open('result.json', 'w') as fp:
	json.dump(results[0][5], fp)

clusters = [[] for _ in range(n_clusters)]
for key in results[0][5].keys():
	clusters[results[0][5][key]].append(key)

for i, cluster in enumerate(clusters):
	print("Elementos Cluster " + str(i))
	print(cluster)