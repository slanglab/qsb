'''Take all of the results and make a big table'''

# scp ahandler@server:/pathTo/bottom_up_clean/*.csv bottom_up_clean/

import csv
from collections import defaultdict

results_fn = "bottom_up_clean/results.csv"

timer_fn = "bottom_up_clean/timer.csv"

results = []

method2slormu = defaultdict(float)
method2slor_std = defaultdict(float)
method2f1 = defaultdict(float)
method2time_mu = defaultdict(float)
method2time_sigma = defaultdict(float)

print_method = {}
print_method["make_decision_lr"] = "Additive"
print_method["only_locals"] = "Additive {\\small (edge only) }"
print_method["make_decision_random"] = "Random {\\small (lower bound) }"
print_method["ilp"] = "ILP"


with open(results_fn, "r") as inf:
    reader = csv.reader(inf)
    header = next(reader)
    for ln in reader:
        f1, slor_mu, slor_std, method = ln
        method2f1[method] = float(f1)
        method2slormu[method] = float(slor_mu)
        method2slor_std[method] = float(slor_std)


with open(timer_fn, "r") as inf:
    reader = csv.reader(inf)
    header = next(reader)
    for ln in reader:
        time_mu, time_sigma, method = ln
        method2time_mu[method] = float(time_mu) * 1000 
        method2time_sigma[method] = float(time_sigma)


def todec(float_):
    dec = "{:.3f}".format(float_)
    small = "{\small " + dec + "}"
    return small


def todec_bold(float_):
    dec = "{:.3f}".format(float_)
    small = "\\textbf{\small " + dec + "}"
    return small


for method in ['make_decision_random', 'ilp', 'only_locals', 'make_decision_lr']:
    if method != "make_decision_lr":
        print("&".join([print_method[method],
              todec(method2f1[method]),
              todec(method2slormu[method]),
              todec(method2time_mu[method]),
              ]) + "\\\\")
    else:
        print("&".join(["\\textbf{" + print_method[method] + "}",
              todec_bold(method2f1[method]),
              todec_bold(method2slormu[method]),
              todec_bold(method2time_mu[method]),
              ]) + "\\\\")
