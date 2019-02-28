import json
import csv
import glob
scores = {}
best_epochs = {}


out = []
experiments = "/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/*"

configs = {}

for experiment in glob.glob(experiments):
    for metric_epoch in glob.glob(experiment + "/" + "*metrics_epoch*"):
        stats = []
        with open(metric_epoch, "r") as inf:
            dt = json.load(inf)
            epoch_metric = float(dt["validation_accuracy"])
            epoch = dt["epoch"]
            print(epoch_metric, epoch)
        with open(experiment + "/config.json", "r") as inf2:
            cf = json.load(inf2)
        #print(cf["train_data_path"])

        try:
            datasize = cf["train_data_path"].split("_").pop().replace('.jsonl', '')
        except:
            if cf["train_data_path"] == "/mnt/nfs/work1/brenocon/ahandler/qsr/lstm_train_3way.jsonl":
                datasize = 1000000
            else:
                datasize = 100000
        stats.append(experiment)
        stats.append(epoch)
        stats.append(cf["iterator"]["batch_size"])
        stats.append(cf["model"]["abstract_encoder"]["num_layers"])
        try:
            stats.append(cf["model"]["abstract_encoder"]["dropout"])
        except:
            print(experiment)
            stats.append(0)
        stats.append(cf["model"]["abstract_encoder"]["hidden_size"])
        stats.append(cf["model"]["classifier_feedforward"]["dropout"])
        stats.append(cf["trainer"]["optimizer"]["weight_decay"])
        stats.append(cf["trainer"]["optimizer"]["lr"])
        stats.append(cf['model']['abstract_encoder']['input_size'])
        stats.append(datasize)
        stats.append(epoch_metric)
        out.append(stats)

out.sort(key=lambda x:x[-1], reverse=True)


with open("tuning/tuner.csv", "w") as of:
    csv_writer = csv.writer(of)

    csv_writer.writerow(["experiment", "epoch", "batch_size", "num_layers",
                         "dropout", "hidden_size", "dropout_ff",
                         "weight_decay", "lr", "embed_dim", "activation", "datasize", "score"])

    csv_writer.writerows(out)
