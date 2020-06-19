# molexplain

Install conda environment (assumes CUDA10.1 compatible driver is available):

```bash
conda env create -f environment.yml
```

Download AZ & CYP models:

``` bash
cd molexplain
wget https://polybox.ethz.ch/index.php/s/dDDMzi3rTbqkWOV/download -O models.tar.gz
tar -xf models.tar.gz
```

Or download processed data and train the model yourself:

```bash
cd molexplain
wget https://polybox.ethz.ch/index.php/s/K0orABbeJmwOUEh/download -O data.tar.gz
tar -xf data.tar.gz
```
