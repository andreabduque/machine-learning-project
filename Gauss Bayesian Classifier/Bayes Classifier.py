import numpy as np
import pandas as pd
from math import exp

d=18
data = pd.read_csv('segmentation1.csv')
data_x = data.drop(axis=1, columns = ["CLASS"]) # <-- novo, é isso que tu queria né?
# data_x = pd.read_csv('x.csv') <-- antigo
classes = ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"]
c = len(classes)

covariance = np.zeros((c,18,18))
mean = np.zeros((c,18))
inv_covar = np.zeros((c,18,18))
det = np.zeros(c)
p_x_w = np.zeros(c)

x = np.array(data_x.iloc[0]) #primeiro exemplo do dataset

for i,classe in enumerate(classes):
    covariance[i] = np.array(data.loc[data["CLASS"]==classe,"REGION-CENTROID-COL":"HUE-MEAN"].cov())
    mean[i] = np.array(data.loc[data["CLASS"]==classe,"REGION-CENTROID-COL":"HUE-MEAN"].mean())
    inv_covar[i] = np.linalg.inv(covariance[i])
    det[i] = abs(np.linalg.det(inv_covar[i]))
    p_x_w[i] = ((2*3.1415)**(d/2))*(det[i]**0.5)*exp(-0.5*(np.matmul(np.matmul((x-mean[i]).T,inv_covar[i]),(x-mean[i]))))
    
p_x_w/p_x_w.sum() #probabilidades a posteriori

