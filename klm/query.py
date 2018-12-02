from __future__ import division
from sklearn import linear_model

import kenlm
import pickle
import numpy as np

logistic = linear_model.LogisticRegression()

model = kenlm.LanguageModel('nyt.klm')

print "*"

print model.score("the boys")/2
print model.score("boys")

print "*"


print model.score("the rudeness")/2
print model.score("rudeness")
