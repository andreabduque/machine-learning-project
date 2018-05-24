import numpy as np
import pandas as pd
from math import exp
import random
from sklearn.model_selection import StratifiedKFold
import math
# import pdb  pdb.set_trace()


class BayesClassifier:
    
    def __init__(self):
        self.covariance = None
        self.mean = None
        self.classes = None
        return
        
    def parameters(self, data): #Entra DataFrame
        self.classes = data["CLASS"].unique()
        data_x = data.drop(axis=1, columns = ["CLASS"])
        d = len(data_x.iloc[0])
        mean = np.zeros((len(self.classes),d))
        covariance = np.zeros((len(self.classes), d, d))
        for i,classe in enumerate(self.classes):
            classes = np.array(data_x.loc[data["CLASS"]==classe])
            qtd_rows = len(classes)
            mean[i] = data.loc[data["CLASS"]==classe].mean()
            x_k = np.zeros((d,d))
            for x in classes:
                x = np.matrix(x)
                x_k = x_k + np.multiply(x.T,x)
            x_k = x_k/qtd_rows
            mi = np.matrix(mean[i])
            mi_k = np.multiply(mi.T,mi)
            covariance[i] = x_k - mi_k
        self.mean = mean
        self.covariance = covariance
    
    def classify(self, x): #Entra np.array
        d = len(x)
        p_x_w = np.zeros(len(self.classes))
        p_w_x = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            try:
                inv_covar = np.linalg.inv(self.covariance[i] * np.identity(d))
            except:
                inv_covar = np.linalg.pinv(self.covariance[i] * np.identity(d)) 

            left = ((2*math.pi)**-d/2)
            mid = (np.linalg.det(inv_covar)**0.5)

            left_exp = (np.matrix(x)-np.matrix(self.mean[i]))
            product_left = np.matmul(left_exp, inv_covar)
            right_exp = np.matrix(x-self.mean[i]).T
            right = np.dot(product_left, right_exp)
            p_x_w[i] = left*mid*exp(-0.5*right)
        try:
            p_w_x = p_x_w/p_x_w.sum()
        except:
            p_w_x = p_x_w/0.0000001
        return(self.classes[np.argmax(p_w_x)])
    
    def accuracy(self, Test):
        coef = 0
        data_x = Test.drop(axis=1, columns = ["CLASS"])
        for i in Test.index.tolist():
            if self.classify(np.array(data_x.loc[i])) == Test.loc[i,"CLASS"]:
                coef += 1
        return coef/len(Test)    
    
    def KfoldNtimes(self, k, n, data): #k = numero de subconjuntos; n = N times
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))
        y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True)
        skf.get_n_splits(X, y)
        l = []
        for i in range(n):
            media = 0
            for train_index, test_index in skf.split(X, y):
                self.parameters(data.loc[train_index])
                media += self.accuracy(data.loc[test_index])
            l.append(media/k)
        return(l)

df = pd.read_csv('segmentation1.csv')
modelo = BayesClassifier()
print(modelo.KfoldNtimes(10,30,df))