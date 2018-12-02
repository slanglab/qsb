from __future__ import division
from sklearn import linear_model

import kenlm
import pickle
import numpy as np

logistic = linear_model.LogisticRegression()

model = kenlm.LanguageModel('fa.klm')

print("*")

print(model.score("I am a student"))

print("*")

print(model.score("student I a am"))
