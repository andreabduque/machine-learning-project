import numpy as np
import pandas as pd
from math import exp
import random
from sklearn.model_selection import StratifiedKFold

class BayesClassifier:
    
    def __init__(self):
        self.covariance = None
        self.mean = None
        self.classes = None
        return
        
    def parameters(self,data): #Entra DataFrame
        self.classes = data["CLASS"].value_counts().to_dict()
        data_x = data.drop(axis=1, columns = ["CLASS"])
        d = len(data_x.iloc[0])
        self.covariance = np.array(data_x.cov())
        #SHIT
        for i in range(d):
            for j in range(d):
                if i!=j:
                    self.covariance[i,j]=0
        #SHIT
        mean = np.zeros((len(self.classes),d))
        for i,classe in enumerate(self.classes):
            mean[i] = np.array(data.loc[data["CLASS"]==classe].mean())
        self.mean = mean
        return

        # view = np.array(data_x)
        # n = view.shape[0]
        # qtd_columns = view.shape[1]

        # x_k = np.empty((1, n))
        # for i in range(0, n):
        #     x_k[i] = np.cross(view[i], view[i])
        # media_x_k = np.sum(x_k)/n

        # self.covariance = np.zeros((qtd_columns, qtd_columns))
        # self.mean = np.zeros(qtd_columns)
        # for i in range(0, qtd_columns):
        #     self.mean[i] = np.cross(np.mean(view[:, i]), np.mean(view[:, i]))
        #     self.covariance[i][i] = media_x_k - n * self.mean[i]
        
        # return
    
    def classify(self, x): #Entra np.array
        inv_covar = np.linalg.inv(self.covariance)
        d = len(x)
        p_x_w = np.zeros(len(self.classes))
        p_w_x = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            p_x_w[i] = ((2*3.1415)**-d/2)*((np.linalg.det(inv_covar))**0.5)*exp(-0.5*(np.matmul(np.matmul((x-self.mean[i]),inv_covar),(x-self.mean[i])))) 
        p_w_x = p_x_w/p_x_w.sum()
        return(list(self.classes.keys())[np.argmax(p_w_x)])
    
    def accuracy(self,Test): #Entra DataFrame
        coef = 0
        data_x = Test.drop(axis=1, columns = ["CLASS"])
        for i in Test.index.tolist():
            if self.classify(np.array(data_x.loc[i])) == Test.loc[i,"CLASS"]:
                coef += 1
        return coef/len(Test)
    
    def KfoldNtimes(self,k,n,data):
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))
        y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True)
        skf.get_n_splits(X, y)
        l = []
        for i in range(n):
            media = 0
            for train_index, test_index in skf.split(X, y):
                #print("TRAIN:", train_index, "TEST:", test_index)
                self.parameters(df.loc[train_index])
                media += self.accuracy(df.loc[test_index])
            l.append(media/10)
        return(l)

#TESTE
df = pd.read_csv('segmentation1.csv')
modelo = BayesClassifier()
print(modelo.KfoldNtimes(10,30,df))
