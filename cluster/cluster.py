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
		# print("inicializando")
		self.initialize_parameters()
		self.initialize_partitions()
		# print("otimizando")
		self.optimize_partition()
		print("fim algoritmo")

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

		# print("prots iniciais")
		# print(self.prototypes)

		# print("pesos iniciais")
		# print(self.weights)


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
			# print(ite)
			ite += 1
			# clusters = [[] for _ in range(self.c)]
			# for key in self.elements.keys():
			# 	clusters[self.elements[key]].append(key)

			# print("calculando melhores prototipos")
			self.update_prototypes(self.clusters)
			# print("calculando hiper parametros")
			self.update_weights(self.clusters)

			# print("alocando nas particoes")
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

				# if(k in self.elements):
				if(nearest_cluster != self.elements[k]):
					test = 1
					nearest_cluster_old = self.elements[k]
					self.elements[k] = nearest_cluster
					# adicionando no novo cluster
					self.clusters[nearest_cluster].append(k)
					self.item_to_position[nearest_cluster][k] = len(self.clusters[nearest_cluster])-1

					# removendo do antigo cluster
					position = self.item_to_position[nearest_cluster_old].pop(k)
					last_item = self.clusters[nearest_cluster_old].pop()
					if position != len(self.clusters[nearest_cluster_old]):
						self.clusters[nearest_cluster_old][position] = last_item
						self.item_to_position[nearest_cluster_old][last_item] = position


		# print(str(ite) + " iteracoes ate minimo local")

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
	part = Partition(3, view)
	part.run()
	return (part.get_objective_function(), part.get_partition_rand_index(groups_class), part.initial_prototypes)
	

data = pd.read_csv("iris.data", sep=",")
view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]


# <<<<<<< Updated upstream
# best_energy = float("inf")

# tempo1 = time.time()
# for i in range(0, 100):
# 	print("iteracao " + str(i))
# 	part.run()
# 	energy = part.get_objective_function()
# 	if(energy < best_energy):
# 		best_energy = energy
# 		best_energy_ari = part.get_partition_rand_index(groups_class)
# 		best_result = part.get_result()

# tempo2 = time.time()

# print(tempo2-tempo1)

# print("Melhor Energia " + str(2*energy))
# print("ARI " + str(best_energy_ari))
# with open('result.json', 'w') as fp:
# 	json.dump(best_result, fp)
# =======
# results = []
# n_iter = range(100)
# pool = Pool(100)
before = time.time()

with Pool(processes=5) as pool:
	results = pool.map(get_clustering_result, 100*[None]) 

# print(time.time() - before)
print("Tempo " + str(time.time() - before))

results.sort(key=lambda tup: tup[0])

# print(results)

# best_energy = float("inf")
# before = time.time()
# for i in range(0, 100):
# 	# print("iteracao " + str(i))
# 	part.run()
# 	energy = part.get_objective_function()
# 	if(energy < best_energy):
# 		best_energy = energy
# 		best_energy_ari = part.get_partition_rand_index(groups_class)
# 		best_result = part.get_result()

print("Melhor Energia " + str(2*results[0][0]))
print("ARI " + str(results[0][1]))
print("Prototipos")
print(results[0][2])
