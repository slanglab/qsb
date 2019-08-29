## Code and data for Linear-time Sentence Compression under Lexical and Length Constraints (EMNLP '19)


### Summary
Extractive [sentence compression](https://www.isi.edu/~marcu/papers/aaai-stat-sum-00.pdf) shortens a source sentence S to a shorter compression C by removing words from S. 

For instance:

S: **Gazprom** the Russian state gas giant announced a 40 percent increase in the price of natural gas sold to **Ukraine** which is heavily dependent on Russia for its gas supply.

C: **Gazprom** announced a 40 percent increase in the price of gas sold to **Ukraine**.

This repo presents our linear-time, query-focused sentence compression technique. Given a source sentence S and a set of query tokens Q, we produce a C that contains all of the words in Q and is shorter than some character budget b.

Our method is much faster than [ILP-based methods](https://www.jamesclarke.net/media/papers/clarke-lapata-jair2008.pdf), another class of algorithms that can also perform query-focused compression. We describe our method in our companion paper.


### Repo tour

- `bottom_up_clean` code for vertex addition is here
- `code` utilities, such as printers, loggers and significance testers
- `dead_code` old code not in use
- `ilp2013` F & A implementation
- `latex` paper & writing
- `klm` some utilties for computing slor
- `preproc` holds preprocessing code
- `scripts` runs experiments
- `snapshots` ILP weights, learned from training. Committed for replicability b/c ILP training takes days

## Some notes on results in paper

### Timing results (including Fig 3)
 
- `scripts/test_timing_results.sh`
- `scripts/rollup_times.R`
- `scripts/latencies.R`

### Table 2
- The script `make_results_master.ipynb` gets the numbers for this table based on two files: 
    - `bottom_up_clean/results.csv`
    - `bottom_up_clean/all_times_rollup.csv`
- Note: this notebook also runs scripts/latencies.R to make figure 3
- Those results files are created via the script `scripts/test_results.sh`

### Figure 3
- The plot `emnlp/times.pdf` comes from `scripts/latencies.R` 
- `R version 3.4.4 (2018-03-15) -- "Someone to Lean On"`
- `Tidyverse version 1.2.1`

### Neural network params

- The neural net uses models/125249540
- The params of the network are stored in the AllenNLP config file `models/125249540/config.json`

### Pickled paths files

The train/test data is packaged as preproc/*.paths files (for oracle path). These files are created by the preprocessing scripts (`$fab preproc`). These files are actually jsonl but not a priority to rename them. They were once pickled. 

Some of these files are too big to commit directly (even zipped) but split and zipped forms are included in the repo

To remake them from the split/zipped versions run `./scripts/unzip_paths.sh`
