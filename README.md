## Quick Start
unzip datasets

make clean && make

./main percentage dataset

For example:

./main 1.27 Hypergraphs/com-amazon-cmty-hygra 

## Datasets 
We evaluate on public hypergraph benchmarks from SNAP (Stanford Large Network Dataset Collection): 

https://snap.stanford.edu/data/index.html.

We use the same conversion pipeline from Julian Shun's paper: "Practical Parallel Hypergraph Algorithms,” PPoPP’20, Artifact Evaluation". Specifically, we download raw SNAP files and convert them into the Hygra layout using the converters from their artifact:

PPoPP’20 AE repo: https://github.com/jshun/ppopp20-ae
 (see utils/)

Our converters mirror those scripts (same field ordering, indexing, and bipartite incidence layout), so inputs produced with the PPoPP’20 pipeline work directly with this code.

Please download the dataset from SNAP first and then convert into the Hygra layout. Next, save them in the "Hypergraphs" folder in order to run the dataset.


