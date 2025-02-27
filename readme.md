# WORK IN PROGRESS
This repository is currently under construction.
Up-to-date information will be added later. Thank you.

# Dataset
The dataset is based on the Audiology dataset. In case of R1_reduced, 1/4 of the rows was selected randomly. The CSV can be found in the R1_reduced directory.
The repository also contains the original version and a version where 1/2 of the rows was selected randomly. As it turned out, 1/2 was too long for LMMs, so we decided to ude the 1/4 version instead.

# FPG results
The FPG results for each itemsets length are also available in the R1_reduced directory.

# Motivation
The classical FPG approach (eg. in Python) fails on the itemsets of length 5 and higher on this dataset because of information explosion.

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
We experimented with LLMs if they are able to find the frequent itemsets in a similar/better time and with simmilar/better accuracy compared to FPG.

## TBA

# Packages
- pyfim              6.28
- pandas             2.2.3
- pyarc              1.1.4
