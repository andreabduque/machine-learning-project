import numpy as np
import pandas as pd
from math import exp
import random
from sklearn.model_selection import StratifiedKFold
import math
import pdb

class BayesClassifier:
    
    def __init__(self):
        self.covariance = None
        self.mean = None
        self.classes = None
        return
        
    def parameters(self,data): #Entra DataFrame
        # self.classes = data["CLASS"].value_counts().to_dict()
        # c = len(self.classes)
        # data_x = data.drop(axis=1, columns = ["CLASS"])
        # d = len(data_x.iloc[0])
        # self.covariance = np.zeros((d,d))
        # #SHIT
        # for j in range(d):
        #     self.covariance[j,j] = np.array(data.cov())[j,j]
        # #SHIT
        # mean = np.zeros((len(self.classes),d))
        # for i,classe in enumerate(self.classes):
        #     mean[i] = np.array(data.loc[data["CLASS"]==classe].mean())
        # self.mean = mean
        # return


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
        print(type(self.mean),type(self.covariance))


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
        
        d = len(x)
        p_x_w = np.zeros(len(self.classes))
        p_w_x = np.zeros(len(self.classes))
        inv_covar = np.linalg.inv(self.covariance)
        for i in range(len(self.classes)):
            a = ((2*math.pi)**-d/2)
            b = ((np.linalg.det(inv_covar))**0.5)
            c = np.multiply((np.matrix(x)-self.mean[i]),inv_covar[i])
            e = np.matrix(x-self.mean[i]).T
            d = np.multiply(c,e)
            print("-------------")
            print(c.shape)
            print("-------------")
            expo = exp(-0.5*d)
            p_x_w[i] = a*b*expo
            #p_x_w[i] = ((2*math.pi)**-d/2)*((np.linalg.det(inv_covar))**0.5)*exp(-0.5*(np.matmul(np.matmul((x-self.mean[i]),inv_covar),(x-self.mean[i])))) 
        p_w_x = p_x_w/p_x_w.sum()
        return(list(self.classes.keys())[np.argmax(p_w_x)])
    
    def accuracy(self,Test): #Entra DataFrame
        coef = 0
        data_x = Test.drop(axis=1, columns = ["CLASS"])
        for i in Test.index.tolist(): #Retorna lista dos indices de treinamento
            #print(self.classify(np.array(data_x.loc[i])), Test.loc[i,"CLASS"])
            if self.classify(np.array(data_x.loc[i])) == Test.loc[i,"CLASS"]: # Se o classificador classificar corretamente
                coef += 1 #Conta o acerto
        return coef/len(Test) #Numero de acertos pelo número total de classificações
    
    def KfoldNtimes(self,k,n,data): #k = numero de subconjuntos; n = N times
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
df = pd.read_csv('iris.data',sep=',')
#df = pd.read_csv('segmentation1.csv')
modelo = BayesClassifier()
#modelo.parameters(df)
print(modelo.KfoldNtimes(10,30,df))