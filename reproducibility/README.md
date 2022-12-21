# Reproducibility Challenge

As part of our 2nd project for EPFL's CS-433 course, we reproduced the results obtained by SynNet, a ML model for *de novo* synthesizable molecular design proposed by Coley et *al.* in this [ICLR 2022 paper](https://arxiv.org/abs/2110.06389).

Our report is available [here](report.pdf)

## Structure

### Notebooks

We organize the computation in three notebooks :
 - [The Data Analysis](data.ipynb) - Where we explore and interpret the data used for the training
 - [The Execution of the Model](run.ipynb) - Where the model is used to produce our results
 - [Result Analysis](analysis.ipynb) - Where we analyse the results and reduce the elements used in the report

### Helper scripts

To correctly modularize the code, we created multiple _helper_ python scripts :
 - [analysis.py](helpers/analysis.py) - Which contains the script used in the result analysis
 - [file_utils.py](helpers/file_utils.py) - Which contains utility functions to perform various tasks on the data files
 - [loader.py](helpers/loader.py) - That we use to load (and retrieve) datasets
 - [optimize.py](helpers/optimize.py) - Where we modularized the optimization script
 - [paths.py](helpers/paths.py) - Where we define/compute useful paths
 - [preprocessors.py](helpers/preprocessor.py) - That contains the modularized version of the preprocessing scripts
 - [synthesis.py](helpers/synthesis.py) - - Where we modularized the synthesis script

### Results

We produced multiple results, they are available in the results folder with the following structure :

```results/{type}/{dataset}/{model}/```

Where :
 - **type** is either `synthesis` or `optimize`
 - **dataset** used, one of `reachable`, `chembl` or `zinc`
 - **model** used, the checkpoint, either `original` or `trained`

### Data

The `data` folder contains every file used at some point by the process.
