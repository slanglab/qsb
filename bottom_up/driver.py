'''

Run the prelim bottom up experiments

'''

from bottom_up.all import *


from bottom_up.all import EasyAllenNLP

dev = [json.loads(_) for _ in open("dev.jsonl")]
dep_probs = train_from_corpus("dev.jsonl")

slens = get_slens(dev, bottom_up_from_corpus_nops, dep_probs=dep_probs)

nn = EasyAllenNLP(loc="/tmp/681705263")

out = get_features_and_labels("mini.train.jsonl", cutoff=100)

v = DictVectorizer(sparse=False)

#X = v.fit_transform([_["feats"] for _ in out])

y = np.asarray([_["y"] for _ in out])

#clf = LogisticRegression(random_state=0,
#                         solver='lbfgs',
#                         C=1,
#                         multi_class='ovr').fit(X, y)

#clf.score(X, y)


nn_picks = f1_experiment(dev, bottom_up_from_nn, nn=nn)
#random_picks = f1_experiment(dev, bottom_up_compression_random, nada=None)
#corpus_picks = f1_experiment(dev, bottom_up_from_corpus, dep_probs=dep_probs)
#lr = f1_experiment(dev, bottom_up_from_clf, clf=clf, v=v)

print("\n")
print("[*] random")
#print(random_picks)
print("[*] corpus")
#print(corpus_picks)
#print("[*] logistic regression", lr)
