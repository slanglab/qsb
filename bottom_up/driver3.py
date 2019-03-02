from bottom_up.all import get_labels_and_features
from bottom_up.all import get_governor

import json
import pickle
import numpy as np
from bottom_up.all import get_lr
from bottom_up.all import get_dependents
from bottom_up.all import get_f1
from bottom_up.all import featurize_ultra_local
from bottom_up.all import featurize_parent_proposal
from bottom_up.all import featurize_child_proposal
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from code.printers import pretty_print_conl
from bottom_up.all import len_tree

training_paths = [_ for _ in open("train.paths")]
validation_paths = [_ for _ in open("validation.paths")]


vectorizer = DictVectorizer(sparse=True)

train_features, train_labels = get_labels_and_features(training_paths)

X_train = vectorizer.fit_transform(train_features)

y_train = np.asarray(train_labels)

val_features, val_labels = get_labels_and_features(validation_paths)

X_val = vectorizer.transform(val_features)

y_val = np.asarray(val_labels)

clf = LogisticRegression(random_state=0,
                         solver='lbfgs',
                         C=1,
                         multi_class='ovr').fit(X_train, y_train)

print(clf.score(X_val,y_val))
print(clf.score(X_train, y_train))

with open("/tmp/a", "wb") as of:
    pickle.dump((clf, vectorizer), of)