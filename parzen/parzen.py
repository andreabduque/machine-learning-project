import numpy as np
import pandas as pd
import math
import random
import pdb
from sklearn.model_selection import StratifiedKFold


class Parzen:

    def __init__(self, data):
        self.classes = data["CLASS"].unique()

    def parameters(self):
        self.h_vector = self.init_h(3)
        self.h = 0

    def kernel(self, x):
        return (1/((2*math.pi)**(1/2)))*math.exp(-x*x/2)

    def init_h(self, n):
        return [random.random()] + [10*random.random() for i in range(n-1)] # gerado entre 0 e 1 - se mostrou mais eficiente

    def parzen(self, data, x, h):
        p = len(x)
        sum = data.shape[0] * [None]
        for i in range(data.shape[0]):
            prod = []
            for j in range(data.shape[1]):
                prod.append(self.kernel((x[j] - data[i][j])/h))
            sum[i] = np.prod(prod)
        return (1/(data.shape[0]*(h**p)))*np.sum(sum)

    def prob(self, data, x, h):
        classes = data["CLASS"].unique()
        data_x = data.drop(axis=1, columns = ["CLASS"])

        p_w_x = len(classes) * [None]
        for i, classe in enumerate(classes):
            # pdb.set_trace()
            x_classe = np.array(data_x.loc[data["CLASS"]==classe])

            p_w_x[i] = self.parzen(x_classe, x, h)

        return self.classes[np.argmax(p_w_x)]

    def accuracy(self, training, Test, h): #Entra DataFrame
        accuracy = 0
        coef = 0
        data_x = Test.drop(axis=1, columns = ["CLASS"])

        coef = 0
        for i in Test.index.tolist():
            if self.prob(training, np.array(data_x.loc[i]), h) == Test.loc[i,"CLASS"]:
                coef += 1

        if((coef/len(Test)) > accuracy):
            accuracy = coef/len(Test)
            
        return accuracy

    def estimate_h(self, data, k = 5, n = 1): #k = numero de subconjuntos; n = N times
        _x = np.array(data.drop(axis=1, columns = ["CLASS"]))
        _y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True) #Classe que Andrea achou que realiza o "K Fold N times"
        skf.get_n_splits(_x, _y)

        train_index =[]
        test_index = []
        for i, j in skf.split(_x, _y):
            train_index = i
            test_index = j
            break

        # for index in test_index:
        self.parameters()
        
        accuracy = 0
        for h in self.h_vector:
            print(h)
            current = self.accuracy(data.iloc[train_index], data.iloc[test_index], h)
            if(current > accuracy):
                accuracy = current
                self.h = h

        return #Soma das acurácias de cada subconjunto de teste

    def KfoldNtimes(self, data, k, n): #k = numero de subconjuntos; n = N times
        X = np.array(data.drop(axis=1, columns = ["CLASS"]))
        y = np.array(data["CLASS"])
        skf = StratifiedKFold(n_splits=k,shuffle=True) #Classe que Andrea achou que realiza o "K Fold N times"
        skf.get_n_splits(X, y)
        l = []

        for i in range(n):
            media = 0
            for train, test in skf.split(X, y):  #retorna duas listas:
                data_train = data.loc[train]
                data_test = data.loc[test]
                self.estimate_h(data_train)
                media += self.accuracy(data_train, data_test, self.h) #Soma das acurácias de cada subconjunto de teste
                l.append(media/k) #média de cada i-Times
        return(l)

#TESTE
df = pd.read_csv('segmentation1.csv')
# df = pd.read_csv('iris.data')
modelo = Parzen(df)
# k = modelo.estimate_h(df)
k = modelo.KfoldNtimes(df, 10,30)
print("------------")
print(k)