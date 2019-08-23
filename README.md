## Supporting code for our short paper "Query-focused Sentence Compression in Linear Time" (EMNLP, 2019), Abram Handler and Brendan O'Connor 

### Summary
Extractive [sentence compression](https://www.isi.edu/~marcu/papers/aaai-stat-sum-00.pdf) shortens a source sentence S to a shorter compression C by removing words from S. 

For instance:

S: **Gazprom** the Russian state gas giant announced a 40 percent increase in the price of natural gas sold to **Ukraine** which is heavily dependent on Russia for its gas supply.

C: **Gazprom** announced a 40 percent increase in the price of gas sold to **Ukraine**.

This repo presents our linear-time, query-focused sentence compression technique. Given a source sentence S and a set of query tokens Q, we produce a C that contains all of the words in Q and is shorter than some character budget b.

Our method is much faster than [ILP-based methods](https://www.jamesclarke.net/media/papers/clarke-lapata-jair2008.pdf), another class of algorithms that can also perform query-focused compression. We describe our method in our companion paper.


##### Repo tour

- `bottom_up_clean` code for vertex addition is here
- `code` utilities, such as printers, loggers and significance testers
- `dead_code` old code not in use
- `ilp2013` F & A implementation
- `latex` paper & writing
- `klm` some utilties for computing slor
- `preproc` holds preprocessing code
- `scripts` runs experiments
- `snapshots` ILP weights, learned from training. Committed for replicability b/c ILP training takes days

##### Some notes

### Timing results
 
- `scripts/test_timing_results.sh`

### Table 2 
    - The script `bottom_up_clean/make_results_table.py` gets the numbers for this table based on two files: 
        - `bottom_up_clean/results.csv`
        - `bottom_up_clean/timer.csv`
    - Those files are created via the script `scripts/test_results.sh`

    - The neural net uses models/125249540

### Figure 3
- The plot `emnlp/times.pdf` comes from `scripts/latencies.R` 
- `R version 3.4.4 (2018-03-15) -- "Someone to Lean On"`
- `Tidyverse version 1.2.1`

### Neural network params

- The params of the network are stored in the AllenNLP config file `models/125249540/config.json`
