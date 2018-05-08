import numpy as np
import pandas as pd
from math import exp
import random

class BayesClassifier:
    
    def __init__(self,data):
        self.data = data
        self.classes = data["CLASS"].value_counts().to_dict()
        return
        
    def parameters(self):
        data_x = self.data.drop(axis=1, columns = ["CLASS"])
        d = len(data_x.iloc[0])
        self.covariance = np.array(data_x.cov())
        mean = np.zeros((len(self.classes),d))
        for i,classe in enumerate(self.classes):
            mean[i] = np.array(self.data.loc[self.data["CLASS"]==classe].mean())
        self.mean = mean
        return
    
    def classify(self, x):
        m,cov = self.mean, self.covariance
        inv_covar = np.linalg.inv(cov)
        g_x = np.zeros(len(self.classes))
        p_c = np.zeros(len(self.classes))
        for i,classe in enumerate(self.classes):
            p_c[i] = self.classes[classe]/len(self.data)
            g_x[i] = (-0.5*(np.matmul(np.matmul((x-m[i]),inv_covar),(x-m[i])))) + np.log(p_c[i]) 
        return(list(self.classes.keys())[np.argmax(g_x)])

#TESTE IN SAMPLE
df = pd.read_csv('segmentation1.csv')
df_x = df.drop(axis=1, columns = ["CLASS"])

model = BayesClassifier(df)
model.parameters()

coef = 0
for i in range(len(df)):
    if model.classify(np.array(df_x.loc[i])) == df.loc[i,"CLASS"]:
        coef += 1
print(coef/len(df))