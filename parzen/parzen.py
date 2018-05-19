import numpy as np
import pandas as pd
import math
import random
import pdb
from sklearn.model_selection import StratifiedKFold


class Parzen:

    def __init__(self, data):
        self.classes = data["CLASS"].unique()

        return

    def parameters(self, data):
        self.h = 0
        self.h_vector = self.init_h(5)
        self.data = data
        self.n = data.shape[0]
        self.p = data.shape[1]

    def kernel(self, x):
        return (1/((2*math.pi)**(1/2)))*math.exp(-x*x/2)

    def init_h(self, n):
        return [5*random.random() for i in range(n)] # gerado entre 0 e 1 - se mostrou mais eficiente

    def parzen(self, data, x, h):
        sum = data.shape[0] * [None]
        for i in range(data.shape[0]):
            prod = []
            for j in range(data.shape[1]):
                prod.append(self.kernel((x[j] - data[i][j])/h))
            sum[i] = np.prod(prod)
        
        return (1/(self.n*(h**7)))*np.sum(sum)

    def prob(self, data, x, h):
        classes = data["CLASS"].unique()
        data_x = data.drop(axis=1, columns = ["CLASS"])

        p_w_x = len(classes) * [None]
        for i, classe in enumerate(classes):
            # pdb.set_trace()
            x_classe = np.array(data_x.loc[data["CLASS"]==classe])

            p_w_x[i] = self.parzen(x_classe, x, h)

        return self.classes[np.argmax(p_w_x)]

    def accuracy(self, training, Test): #Entra DataFrame
        accuracy = 0
        coef = 0
        data_x = Test.drop(axis=1, columns = ["CLASS"])
        for h in self.h_vector:
            print(h)
            coef = 0
            for i in Test.index.tolist():
                if self.prob(training, np.array(data_x.loc[i]), h) == Test.loc[i,"CLASS"]:
                    coef += 1

            if((coef/len(Test)) > accuracy):
                accuracy = coef/len(Test)
                self.h = h
            
        return accuracy

    def estimate_h(self, data, k = 5, n = 1): #k = numero de subconjuntos; n = N times
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))
        y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True) #Classe que Andrea achou que realiza o "K Fold N times"
        skf.get_n_splits(X, y)
        l = []
        media = 0
        train_index =[]
        test_index = []
        for i, j in skf.split(X, y):
            train_index = i
            test_index = j
            break

        # for index in test_index:
        self.parameters(data.loc[train_index])
            # self.prob(data.loc[train_index], data.loc[index]) #Estimação dos parametros da distribuição normal a partir dos dados de treinamento
        return(self.accuracy(data.loc[train_index], data.loc[test_index])) #Soma das acurácias de cada subconjunto de teste
        #     l.append(media/len(test_index)) #média de cada i-Times
        # return(l)

    def KfoldNtimes(self, data, k = 5, n = 1): #k = numero de subconjuntos; n = N times
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))
        y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True) #Classe que Andrea achou que realiza o "K Fold N times"
        skf.get_n_splits(X, y)
        l = []
        for i in range(n):
            media = 0
            train_index, test_index = skf.split(X, y)[0] #retorna duas listas:
            for index in test_index:
                self.parameters(data.loc[train_index])
                self.prob(data.loc[train_index], data.loc[index]) #Estimação dos parametros da distribuição normal a partir dos dados de treinamento
                media += self.accuracy(data.loc[test_index]) #Soma das acurácias de cada subconjunto de teste
            l.append(media/k) #média de cada i-Times
        return(l)

#TESTE
df = pd.read_csv('segmentation1.csv')
# df = pd.read_csv('iris.data')
modelo = Parzen(df)
k = modelo.estimate_h(df)
print("------------")
print(k)