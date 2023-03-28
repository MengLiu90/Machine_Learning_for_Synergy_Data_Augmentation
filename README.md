# Synergy-Data-Augmentation
This repo tests the augmented synergy data with random forest and gradient boosting trees (GBT). 
These two models are trained against both the original synergy data from AZ-DREAM challenge (https://www.synapse.org/#!Synapse:syn4231880/wiki/) and
the augmented synergy data. The performance of the models on different datasets is compared to investigate the ability of the augmented data in improving the predictive accuracy of machine learning models.
# Dependencies
1. pandas
2. numpy
3. sklearn
4. matplotlib
5. tarfile
# Usage
To reproduce the results in the paper, simply download the repository and run the .py files. <br />
```python RandomForest_random_split.py```<br />
```python RandomForest_tissue_split.py.py```<br />
```python GBT_random_split.py```<br />
```python GBT_tissue_split.py```<br />
```python Code_for_ROC_plots.py```<br />
# Datasets
original_synergy_data.tgz file provides the original synergy data. The complete augmented data can be accessed through https://lsu.box.com/v/data-augmentation
## Cross-validation
## Data splitting method
