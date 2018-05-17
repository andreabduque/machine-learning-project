import numpy as np
import pandas as pd
from math import exp
import random
from sklearn.model_selection import StratifiedKFold
import math
import pdb # pdb.set_trace()

class BayesClassifier:
    
    def __init__(self):
        self.covariance = None
        self.mean = None
        self.classes = None
        return
        
    def parameters(self, data): #Entra DataFrame
        self.classes = data["CLASS"].value_counts().to_dict()
        data_x = data.drop(axis=1, columns = ["CLASS"])
        d = len(data_x.iloc[0])
        mean = np.zeros((len(self.classes),d))
        covariance = np.zeros((len(self.classes), d, d)) # melhorar isso dinamicamente
        for i,classe in enumerate(self.classes):
            # pdb.set_trace()
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
            covariance[i] = x_k - qtd_rows*mi_k
        self.mean = mean
        self.covariance = covariance
    
    def classify(self, x): #Entra np.array
        d = len(x)
        p_x_w = np.zeros(len(self.classes))
        p_w_x = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            inv_covar = np.linalg.inv(self.covariance[i]) * np.identity(d)

            left = ((2*math.pi)**-d/2)
            mid = (np.linalg.det(inv_covar)**0.5)

            left_exp = (np.matrix(x)-np.matrix(self.mean[i]))
            product_left = np.matmul(left_exp, inv_covar)
            right_exp = np.matrix(x-self.mean[i]).T
            right = np.dot(product_left, right_exp)
            
            p_x_w[i] = left*mid*exp(-0.5*right)
            #p_x_w[i] = ((2*math.pi)**-d/2)*((np.linalg.det(inv_covar))**0.5)*exp(-0.5*(np.matmul(np.matmul((x-self.mean[i]),inv_covar),(x-self.mean[i])))) 
        p_w_x = p_x_w/p_x_w.sum()
        return(list(self.classes.keys())[np.argmax(p_w_x)])
    
    def accuracy(self, Test): #Entra DataFrame
        coef = 0
        data_x = Test.drop(axis=1, columns = ["CLASS"])
        for i in Test.index.tolist(): #Retorna lista dos indices de treinamento
            #print(self.classify(np.array(data_x.loc[i])), Test.loc[i,"CLASS"])
            if self.classify(np.array(data_x.loc[i])) == Test.loc[i,"CLASS"]: # Se o classificador classificar corretamente
                coef += 1 #Conta o acerto
        return coef/len(Test) #Numero de acertos pelo número total de classificações
    
    def KfoldNtimes(self, k, n, data): #k = numero de subconjuntos; n = N times
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))
        y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True) #Classe que Andrea achou que realiza o "K Fold N times"
        skf.get_n_splits(X, y)
        l = []
        for i in range(n):
            media = 0
            for train_index, test_index in skf.split(X, y): #retorna duas listas:
            #train_index é uma lista com os índices dos exemplos que serão usados no treinamento
            #test_index é uma lista com os índices usado para verificar a quantidade de acertos do classificador
                #print("TRAIN:", train_index, "TEST:", test_index)
                self.parameters(data.loc[train_index]) #Estimação dos parametros da distribuição normal a partir dos dados de treinamento
                media += self.accuracy(data.loc[test_index]) #Soma das acurácias de cada subconjunto de teste
            l.append(media/k) #média de cada i-Times
        return(l)

#TESTE
df = pd.read_csv('iris.data')
#df = pd.read_csv('segmentation1.csv')
modelo = BayesClassifier()
#modelo.parameters(df)
print(modelo.KfoldNtimes(10,30,df))