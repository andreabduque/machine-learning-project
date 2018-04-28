import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import adjusted_rand_score
import functions as f
import json

class Partition:
	def __init__(self, nclusters, view):
		self.c = nclusters
		self.view = view

	def run(self):
		# print("inicializando")
		self.initialize_parameters()
		self.initialize_partitions()
		# print("otimizando")
		self.optimize_partition()

	def get_objective_function(self):
		clusters = [[] for _ in range(self.c)]
		for key in self.elements.keys():
			clusters[self.elements[key]].append(key)

		energy = 0
		for i in range(0, self.c):
			for k in range(len(clusters[i])):
				energy += 2*(1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[clusters[i][k]]),
					np.array(self.prototypes[i])))

		return energy


	def initialize_parameters(self):
		#Number of variables
		self.p = len(self.view.columns)
		#A suitable parameter
		self.gamma = (1/f.sigma_squared(self.view.as_matrix()))**self.p
		#Global weights for each variable
		self.weights = self.p*[None]
		#Initialize weights
		for i in range(0, self.p):
			self.weights[i] = self.gamma**(1/self.p)

		#Set initial random prototypes
		random_sample = self.view.sample(frac=1)[0:7]
		self.prototypes = random_sample.as_matrix()
		self.initial_prototypes = random_sample.index.values

		# print("prots iniciais")
		# print(self.prototypes)

		# print("pesos iniciais")
		# print(self.weights)


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

	#Compute best prototypes and best hyper parameters
	#Returns partition energy
	def optimize_partition(self):
		test = 1
		ite = 1
		while(test):
			# print(ite)
			ite += 1
			clusters = [[] for _ in range(self.c)]
			for key in self.elements.keys():
				clusters[self.elements[key]].append(key)

			# print("calculando melhores prototipos")
			self.update_prototypes(clusters)
			# print("calculando hiper parametros")
			self.update_weights(clusters)

			# print("alocando nas particoes")
			test = 0
			for k in range(0, len(self.view)):
				dist = float("inf")
				nearest_cluster = 0
				#Kernel between element and cluster prototype
				for h in range(0, self.c):
					x = 2*(1 - f.gaussian_kernel(self.weights, np.array(self.view.iloc[k]),
						np.array(self.prototypes[h])))
					if (x < dist):
						nearest_cluster = h
						dist = x

				if(k in self.elements):
					if(nearest_cluster != self.elements[k]):
						test = 1

				self.elements[k] = nearest_cluster

		print(str(ite) + " iteracoes ate minimo local")

	def update_prototypes(self, clusters):
		for i in range(0, self.c):
			#Do not fool yourself: Numerator is a vector
			num  =  0
			denom = 0
			for j in range(0, len(clusters[i])):
				kernel = f.gaussian_kernel(self.weights, self.view.iloc[clusters[i][j]], self.prototypes[i])
				num   += kernel * self.view.iloc[clusters[i][j]]
				denom += kernel

			if(denom != 0):
				self.prototypes[i] = num / denom

	def update_weights(self, cluster):
		new_weights = self.p*[None]

		for j in range(0, self.p):
			# print("otimizando parametro " + str(j) + " de " + str(self.p))
			produtorio = 1
			denominador = 0
			flag = 1

			for h in range(0, self.p):
				numerador = 0
				for i in range(0, self.c):
					for k in range(0, len(cluster[i])):
						#An Element from cluster
						x_k = np.array(self.view.iloc[cluster[i][k]])
						#Prototype from cluster
						g_i =  self.prototypes[i]
						#Numerator
						parcela_gaussiana = f.gaussian_kernel(self.weights, x_k, g_i)
						segunda_parcela_numerador = (x_k[h] - g_i[h])**2
						numerador += parcela_gaussiana * segunda_parcela_numerador

						#Denominator
						if(flag == 1):
							segunda_parcela_denominador = (x_k[j] - g_i[j])**2
							denominador += parcela_gaussiana * segunda_parcela_denominador
				flag = 0
				produtorio *= numerador

			produto = (self.gamma**(1/self.p)) * (produtorio **(1/self.p))
			new_weights[j] = produto/denominador

		# print("pesos atualizados")
		# print(new_weights)
		self.weights = new_weights

	def get_result(self):
		return self.elements

	def get_partition_rand_index(self, true_labels):
		pred_labels = len(self.view)*[None]
		for i, key in enumerate(sorted(self.elements)):
			pred_labels[i] = self.elements[key]

		return adjusted_rand_score(true_labels, pred_labels)

data = pd.read_csv("iris.data", sep=",")
view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]

part = Partition(3, view)

best_energy = float("inf")
for i in range(0, 100):
	print(str(i))
	part.run()
	energy = part.get_objective_function()
	if(energy < best_energy):
		best_energy = energy
		best_energy_ari = part.get_partition_rand_index(groups_class)
		best_result = part.get_result()

print("Melhor Energia " + str(energy))
print("ARI " + str(best_energy_ari))
with open('result.json', 'w') as fp:
	json.dump(best_result, fp)
