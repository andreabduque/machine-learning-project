import sys 
sys.path.append("../parzen")
sys.path.append("../Gauss")

import numpy as np
import pandas as pd
from parzen import Parzen
from BayesClassifier import BayesClassifier
from sklearn.model_selection import StratifiedKFold

segmentation = pd.read_csv("../segmentation1.csv")
shape_view = pd.read_csv("../shape_view.csv")
rgb_view = pd.read_csv("../rgb_view.csv")

classes = segmentation["CLASS"].unique()

bayes_1 = BayesClassifier()
bayes_2 = BayesClassifier()
bayes_3 = BayesClassifier()

parzen_1 = Parzen(segmentation)
parzen_2 = Parzen(shape_view)
parzen_3 = Parzen(rgb_view)

def split_views(a):
	b = a.loc[:, "REGION-CENTROID-COL": "HEDGE-SD"]
	b["CLASS"] = a["CLASS"]
	c = a.loc[:, "INTENSITY-MEAN": ]
	c["CLASS"] = a["CLASS"]
	return b, c


def att_Bayes(completeView):
	View2, View3 = split_views(completeView)

	bayes_1.parameters(completeView)
	bayes_2.parameters(View2)
	bayes_3.parameters(View3)
	return

def att_Parzen(completeView):
	View2, View3 = split_views(completeView)

	parzen_1.estimate_h(completeView)
	parzen_2.estimate_h(View2)
	parzen_3.estimate_h(View3)
	return

def sum_rule(x, completeView):
	x2 = x[:8]
	x3 = x[8:]
	View2, View3 = split_views(completeView)

	bayes_1.classify(x)
	bayes_2.classify(x2)
	bayes_3.classify(x3)
	parzen_1.prob(completeView, x, parzen_1.h)
	parzen_2.prob(View2, x2, parzen_2.h)
	parzen_3.prob(View3, x3, parzen_3.h)

	bayes_sum = bayes_1.p_w_x + bayes_2.p_w_x + bayes_3.p_w_x
	parzen_sum = parzen_1.p_w_x + parzen_2.p_w_x + parzen_3.p_w_x

	return classes[np.argmax(bayes_sum + parzen_sum)]

def accuracy(Train, Test):
	accuracy = 0
	classe = Test.CLASS
	test = np.array(Test.drop(axis=1, columns = ["CLASS"]))

	for i, x in enumerate(test):
		if  sum_rule(x, Train) == classe.iloc[i]:
			accuracy += 1
	print(accuracy/len(Test))
	return accuracy/len(Test)

def KfoldNtimes(data, k, n): #k = numero de subconjuntos; n = N times
    X = np.array(data.drop(axis=1, columns = ["CLASS"]))
    y = np.array(data["CLASS"])
    l = []

    for i in range(n):
        media = 0
        flag = 0
        skf = StratifiedKFold(n_splits=k,shuffle=True)
        skf.get_n_splits(X, y)
        for train, test in skf.split(X, y):  #retorna duas listas:
            data_train = data.loc[train]
            data_test = data.loc[test]
            att_Bayes(data_train)
            if flag == 0:
                att_Parzen(data_train)
                flag = 1
            media += accuracy(data_train, data_test)
        print(media/k)
        l.append(media/k) #media de cada i-Times
    return(l)

print(KfoldNtimes(segmentation, 10, 30))