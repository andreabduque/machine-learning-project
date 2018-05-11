import numpy as np
import pandas as pd
import math
import random
import pdb

class Parzen:

    def __init__(self, data):
        self.h = self.init_h(data)[4] # metade do array
        self.data = data
        self.n = data.shape[0]
        self.p = data.shape[1]

        return

    def kernel(self, x):
        return (1/((2*math.pi)**(1/2)))*math.exp(-x*x/2)

    def init_h(self, data):
        return [0.04*int(10*random.random()) for i in range(10)] # estudarcomo melhorara a convergencia ~ em função do conjunto de validação

    def parzen(self, data, x):
        sum = self.n * [None]
        for i in range(self.n):
            prod = self.p * [None]
            for j in range(self.p):
                prod[j] = self.kernel((x[j] - data[i][j])/self.h)
            sum[i] = np.prod(prod)
        
        return (1/(self.n*(self.h**7)))*np.sum(sum)

#TESTE
df = pd.read_csv('segmentation1.csv')
modelo = Parzen(np.array(df.drop(axis=1, columns = ["CLASS"])))
print(modelo.parzen(modelo.data, modelo.data[0])) # teste zoado