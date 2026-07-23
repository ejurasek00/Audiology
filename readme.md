# WORK IN PROGRESS
This repository is currently under construction.
Up-to-date information will be added later. Thank you.

# Dataset
The first dataset is based on the Audiology dataset. In the case of R1_reduced, 1/4 of the rows were selected randomly using a random number generator on Mockaroo. The CSV can be found in the R1_reduced directory.
The repository also contains the original version and a version where 1/2 of the rows were selected randomly. As it turned out, the 1/2 version was too long for LLMs, so we decided to use the 1/4 version instead.

The second set of datasets was hand-crafted. The purpose was to create a dataset where there are multiple values in the columns that are the same, and their amount is not too high.
There are 4 modifications:
- Unsorted (50x5)
- Unsorted-mid (75x5)
- Semi-sorted (50x5, sorted by the Temperature column)
- Extended (75x8)

# FPG results
The FPG/Apriori results for each itemsets length are also available in each directory.

# Motivation
The classical FPG approach (e.g., in Python) fails on the itemsets of length 5 and higher on this dataset because of information explosion in the case of the Audiology dataset that was reduced to 1/4.

The FPG was tested on these configurations:

Laptop
- Intel i7-10510U (8) @ 4.900GHz CPU
- Intel CometLake-U GT2 GPU
- 15695 MiB RAM
- GNU/Linux

Jupyter environment 1
-  (32) CPU
-  GPU
- 7168 MiB RAM
- GNU/Linux

# Proposed methods
We experimented with LLMs to see if they are able to find the frequent itemsets in a similar/better time and with similar/better accuracy compared to FPG.
The current aim is to make the process automatic.

# Packages
- pyfim              6.28
- pandas             2.2.3
- pyarc              1.1.4
