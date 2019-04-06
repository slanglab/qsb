Extractive [sentence compression](https://www.isi.edu/~marcu/papers/aaai-stat-sum-00.pdf) shortens a source sentence S to a shorter compression C by removing words from S. 

For instance:

S: **Gazprom** the Russian state gas giant announced a 40 percent increase in the price of natural gas sold to **Ukraine** which is heavily dependent on Russia for its gas supply.

C: **Gazprom** announced a 40 percent increase in the price of gas sold to **Ukraine**.

This repo presents our linear-time, query-focused sentence compression technique. Given a source sentence S and a set of query tokens Q, we produce a C that contains all of the words in Q and is shorter than some character budget b.

Our method is much faster than [ILP-based methods](https://www.jamesclarke.net/media/papers/clarke-lapata-jair2008.pdf), another class of algorithms that can also perform query-focused compression. We describe our method in our companion paper.