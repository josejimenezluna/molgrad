# molexplain

Install conda environment (assumes CUDA10.1 compatible driver is available):

```bash
conda env create -f environment.yml
```


Download publicly available data:

```bash
cd molexplain
wget https://polybox.ethz.ch/index.php/s/m1xwgbyPz4uHwYS/download -O data.tar.gz
tar -xf data.tar.gz

```


Download trained models:

``` bash
wget https://polybox.ethz.ch/index.php/s/5KrmRzDsHfsgfYJ/download -O models.tar.gz
tar -xf models.tar.gz
```
