import numpy as np
import pandas as pd
import math
import random
import pdb

class Parzen:

    def __init__(self, data):
        # self.h = self.init_h(data)[4] # metade do array
        self.h = 0.2
        self.data = data
        self.n = data.shape[0]
        self.p = data.shape[1]

        return

    def kernel(self, x):
        return (1/((2*math.pi)**(1/2)))*math.exp(-x*x/2)

    def init_h(self, data):
        return [0.04*int(10*random.random()) for i in range(10)] # estudarcomo melhorara a convergencia ~ em função do conjunto de validação

    def parzen(self, data, x):
        sum = data.shape[0] * [None]
        for i in range(data.shape[0]):
            prod = data.shape[1] * [None]
            for j in range(data.shape[1]):
                prod[j] = self.kernel((x[j] - data[i][j])/self.h)
            sum[i] = np.prod(prod)
        
        return (1/(self.n*(self.h**7)))*np.sum(sum)

    def prob(self, data, x):
        self.classes = data["CLASS"].value_counts().to_dict()
        data_x = data.drop(axis=1, columns = ["CLASS"])

        p_w_x = len(self.classes) * [None]
        for i, classe in enumerate(self.classes):
            # pdb.set_trace()
            classe = np.array(data_x.loc[data["CLASS"]==classe])

            p_w_x[i] = self.parzen(classe, x)

        return np.argmax(p_w_x)

#TESTE
df = pd.read_csv('segmentation1.csv')
modelo = Parzen(np.array(df.drop(axis=1, columns = ["CLASS"])))
print(modelo.prob(df, modelo.data[0])) # teste zoado