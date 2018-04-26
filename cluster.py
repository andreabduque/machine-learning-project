import pandas as pd
import numpy
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv("segmentation.csv", sep=",")


def euclidean_matrix(view):


def get_prototypes(nprototypes, view):
	gamma = 1

	print(view.sample(frac=1)[0:7])


def set_initial_parameters():


view = data.drop(axis=1, columns = ["CLASS"])
groups_class = data["CLASS"]
#get_prototypes(7, data)

#primeiro elemento
print(np.array(view.iloc[0]))