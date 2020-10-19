# molexplain

Install conda environment (assumes CUDA10.1 compatible driver is available):

```bash
conda env create -f environment.yml
```


Download publicly available data:

```bash
cd molexplain
wget https://polybox.ethz.ch/index.php/s/K0orABbeJmwOUEh/download -O data.tar.gz
tar -xf data.tar.gz

```


Download trained graph neural network models:

``` bash
wget https://polybox.ethz.ch/index.php/s/5KrmRzDsHfsgfYJ/download -O models.tar.gz
tar -xf models.tar.gz
```

Download baseline Sheridan et al. (2019) random forests models:

```
wget https://polybox.ethz.ch/index.php/s/xVa0wX5qnKlTi2J/download -O baseline_models.tar.gz
tar -xf baseline_models.tar.gz

```