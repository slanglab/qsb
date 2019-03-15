'''Take all of the results and make a big table'''

import csv

results_fn = "bottom_up_clean/results.csv"

timer_fn = "bottom_up_clean/timer.csv"

results = []

method2slormu = {}
method2slor_std = {}
method2f1 = {}
method2time_mu = {}
method2time_sigma = {}

with open(results_fn, "r") as inf:
    reader = csv.reader(inf)
    header = reader.next()
    for ln in reader:
        f1, slor_mu, slor_std, method = ln
        method2f1[method] = float(f1)
        method2slormu[method] = float(slor_mu)
        method2slor_std[method] = float(slor_std)


with open(timer_fn, "r") as inf:
    reader = csv.reader(inf)
    header = reader.next()
    for ln in reader:
        time_mu, time_sigma, method = ln
        method2time_mu[method] = float(time_mu)
        method2time_sigma[method] = float(time_sigma)


def todec(float_):
    return "{:.3f}".format(float_)

for method in ['make_decision_lr', 'make_decision_random']:  # ['ilp']
    print(method,
          todec(method2f1[method]),
          todec(method2time_mu[method]),
          todec(method2time_sigma[method]),
          todec(method2slormu[method]),
          todec(method2slor_std[method]))
