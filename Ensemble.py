import numpy as np
import pandas as pd
from parzen import Parzen
from BayesClassifier import BayesClassifier

segmentation = pd.read_csv("segmentation1.csv")
shape_view = pd.read_csv("shape_view.csv")
rgb_view = pd.read_csv("rgb_view.csv")

bayes_1 = BayesClassifier()
bayes_2 = BayesClassifier()
bayes_3 = BayesClassifier()

parzen_1 = Parzen(segmentation)
parzen_2 = Parzen(shape_view)
parzen_3 = Parzen(rgb_view)


def sum_rule(x1, x2, x3, view1, view2, view3):
	bayes_1.parameters(view1)
	bayes_2.parameters(view2)
	bayes_3.parameters(view3)

	parzen_1.estimate_h(view1)
	parzen_2.estimate_h(view2)
	parzen_3.estimate_h(view3)

	bayes_1.classify(x1)
	bayes_2.classify(x2)
	bayes_3.classify(x3)
	parzen_1.estimate_h(view1)
	parzen_2.estimate_h(view2)
	parzen_3.estimate_h(view3)



	return bayes_1.p_w_x + bayes_2.p_w_x + bayes_3.p_w_x




