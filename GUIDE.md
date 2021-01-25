# Description of each file contained in the current repository

In order to better facilitate reproduction of the results presented on the manuscript, as well as allowing others to build upon the presented work, we describe the purpose of each file contained in the repository (as of Jan. 11th 2021)


```
├── environment.yml : contains the specific conda environment used throughout this work 
├── molgrad
│   ├── baseline_models/ : contains baseline trained random forests for comparison against Sheridan's (2019) [1] approach.
│   ├── baseline_prod.py : contains logic for training of the production baseline random-forest models.  
│   ├── baseline_train.py : training code for the baseline random forest models in the datasets considered in this study.
│   ├── baseline_utils.py : utility functions for the coloring fingerprint-based approach described in Sheridan (2019) [1].
│   ├── clean_data.py : cleaning procedure undertaken for the datasets used in each considered endpoint (PPB, Caco-2, hERG & CYP3A4)
│   ├── ig.py : utility functions implementing the integrated gradients procedure detailed in this work.
│   ├── main.py : main runtime file to generate explanations for a set of molecules, using either a provided or a user-trained model. For details on usage check the README.md file
│   ├── models/ : contains trained message-passing models for the endpoints used in this study.
│   ├── net.py : definition of the message-passing neural network used in this study.
│   ├── net_utils.py : utility functions for the featurization and data-feeding routines used for message-passing neural network models.
│   ├── notebooks/ : contains several notebooks exploring individual coloring examples, and global importance analyses (e.g. those presented in the manuscript figures and others)
│   ├── prod.py : contains production utility functions for other files and notebooks.
│   ├── scaffold_oof.py : contains logic regarding necessary generated files for the out-of-fold property cliff analyses.
│   ├── scaffold.py : same as scaffold_oof.py, but for the production models and training datasets used in this work.
│   ├── train_ext.py : contains code to allow users to train message-passing models on their own endpoints. Please check the README.md file for more details on how to do this.
│   ├── train_prod.py : contains logic for the training of the production message-passing neural-network models.
│   ├── train.py : training code for the k-fold cv procedure for the presented message-passing neural network models usd in this study.
│   ├── utils.py : utility functions (e.g. for file handling) used throughout the code.
│   ├── vis_baseline.py : visualization routines for the approach presented by Sheridan (2019) [1]
│   └── vis.py : visualization routines for the approach presented in this work.
└── README.md

```

If further technical information is needed toward replication/application of the presented approach, please feel free to open an issue in the repository, or [send us a mail](mailto:jose.jimenez@rethink.ethz.ch).

## References

[1] Sheridan, Robert P. "Interpretation of QSAR models by coloring atoms according to changes in predicted activity: how robust is it?." Journal of Chemical Information and Modeling 59.4 (2019): 1324-1337.
