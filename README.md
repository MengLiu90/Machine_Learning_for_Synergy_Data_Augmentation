# Synergy-Data-Augmentation
This repo tests the augmented synergy data with random forest and gradient boosting trees (GBT). 
These two models are trained against both the original synergy data from AZ-DREAM challenge (https://www.synapse.org/#!Synapse:syn4231880/wiki/) and
the augmented synergy data. The performance of the models on different datasets is compared to investigate the ability of the augmented data in improving the predictive accuracy of machine learning models.
## Dependencies
1. pandas
2. numpy
3. sklearn
4. matplotlib
5. tarfile
## Usage
To reproduce the results in the paper, simply download the repository and run the .py files. <br />
```python RandomForest_random_split.py```<br />
```python RandomForest_tissue_split.py.py```<br />
```python GBT_random_split.py```<br />
```python GBT_tissue_split.py```<br />
```python Code_for_ROC_plots.py```<br />
## Datasets
The file original_synergy_data.tgz provides the original synergy data used for classification. The complete augmented data can be accessed through https://lsu.box.com/v/data-augmentation. We randomly sampled from the augmented dataset to improve efficiency and reduce memory usage during the training process. The augmented data were only used for training, the models were tested against the original synergy data. 
## Cross-validation
## Data splitting method
