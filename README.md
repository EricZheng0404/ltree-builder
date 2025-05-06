###############################################################
2025 Spring
CS151 Debugging Cloud Computing Project:
Recreate L-tree in LatenSeer Paper
###############################################################

###############################################################
Use
###############################################################
python3 [casper_dir] [--out] [--limit]

- [caper_dir]: path to folder with CASPER CSVs.
- [--out]: output file name for DOT format.
- [--limit]: optional limit on number of CSV files to process

Example Use Case:
python ltree_builder.py /path/to/casper_csvs/ --out ltree.dot --limit 5

###############################################################
Result
###############################################################
The output is a DOT (Graphviz) file representing an L-tree, the tree-strectured
visualization of latency patterns reconstructed from distributed trace data.

- Structure:
  Each line describes a directed edge between two node: a parent node and a child
  node. Then, it comes with a edge label, denoting p50 (median latency of the span)
  and n (how many traces we've observed this span).

###############################################################
Files
###############################################################
