# molgrad

[![DOI](https://zenodo.org/badge/244630479.svg)](https://zenodo.org/badge/latestdoi/244630479)


Supporting code for: Jiménez-Luna _et al._'s "Coloring molecules with explainable artificial intelligence for preclinical relevance assessment", available in [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01344)



## Installation

The recommended method of usage is via the Anaconda Python distribution. One can use one of the provided conda environments in the repository (should work for most *nix systems):

If a CUDA-capable GPU is available, use the `environment.yml` file:

```bash
conda env create -f environment.yml
```

For a CPU-only installation, use the `environment_cpu.yml` file instead:

```bash
conda env create -f environment_cpu.yml
```

To use the graph neural-network models that were trained for the manuscript (plasma protein binding, Caco-2 passive permeability, hERG & CYP3A4 inhibition), you need to download them from:

``` bash
wget https://polybox.ethz.ch/index.php/s/dDDMzi3rTbqkWOV/download -O models.tar.gz
tar -xf models.tar.gz
```

Then activate the environment and prepend the folder to your PYTHONPATH environment variable:

```bash
conda activate molgrad
export PYTHONPATH=/path_to_repo_root/:$PYTHONPATH
```

### (Optional) Download datasets

All the training data used in this study can be freely downloaded from:

```bash
wget https://polybox.ethz.ch/index.php/s/K0orABbeJmwOUEh/download -O data.tar.gz
tar -xf data.tar.gz
```


## Usage

In order to generate explanations for a particular molecule, given a trained model, one only needs to call the `main.py` script. 

```bash
python molgrad/main.py -model_path model_weights.pt -smi SMILES -output_f RESULT_DIR
```


For instance, if we wanted to obtain feature colorings for nicotine for the hERG inhibition pre-trained endpoint, and store it under a home subfolder named `results`, one would do:

```bash
python molgrad/main.py -model_path models/herg_noHs.pt -smi "CN1CCCC1C2=CN=CC=C2" -output_f $HOME/results/
```

This will create a comma-separated file `global.csv` in that folder, with feature attributions corresponding to global variables (_i.e_. molecular weight, log _P_, TPSA, and number of hydrogen donors). Another subfolder `svg` will be created with the produced feature colorings.

Further parameters (such as feeding an entire .smi) for batch prediction and coloring can be checked via the provided help:

```
python molgrad/main.py --help
```

## (Optional) Train your own models:

The current framework also provides functionality for model training using custom data with the `train_ext.py` script. It assumes training data comes in a comma-separated (.csv) file, with one column carrying SMILES and another the target value, whose names need to be specified. For instance:


```bash
python molgrad/train_ext.py -data CSV_FILE -smiles_col "SMILES_COL" -target_col "TARGET_COL" -output path_to_weights.pt
```

The trained model can be then used to color molecules via the `main.py` routine as described above. Additional training options can be consulted with:

```bash
python molgrad/train_ext.py --help
```


## Data collection for XAI model validation

A comma-separated file with examples drawn from the literature to validate this and other XAI approaches can be downloaded from [here](https://polybox.ethz.ch/index.php/s/olEIsl2fPngzFYS).



## Citation

If you use this code (or parts thereof), please use the following BibTeX entry:

```
@article{jimenez2021coloring,
author = {Jiménez-Luna, José and Skalic, Miha and Weskamp, Nils and Schneider, Gisbert},
title = {Coloring Molecules with Explainable Artificial Intelligence for Preclinical Relevance Assessment},
journal = {Journal of Chemical Information and Modeling},
volume = {61},
number = {3},
pages = {1083-1094},
year = {2021},
}
```
