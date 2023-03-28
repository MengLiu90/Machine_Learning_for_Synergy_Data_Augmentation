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
The file ```./Data/original_synergy_data.tgz``` provides the original synergy data used for classification. The complete augmented data can be accessed through https://lsu.box.com/v/data-augmentation. We randomly sampled from the augmented dataset to improve efficiency and reduce memory usage during the training process. The augmented data used are provided in the ```./Data``` folder to reproduce the results in the paper. The augmented data were only used for training, the models were tested against the original synergy data. <br />
<br />
## Training
To train your own models with the augmented data, after splitting the original data into train/test sets, please make sure that only training set-related augmentation data are used to enlarge the training space, otherwise there will be an overlap between the training and testing sets as augmented instances originated from the testing set are included during training.
## Cross-validation
This repo conducts 5-fold cross-validation under random split and tissue-level split.<br />
<br />
Random split is the method widely adopted in machine learning literature, however, it neglects the fact that cell lines from the same tissue can present in both training and testing sets, which can lead to overlap between these two sets as instances involving similar cell lines tend to have comparable feature representations, such as gene expression profiles.<br />
<br />
Tissue-level split partitions the data based on tissues from which the cell lines originate. It can eliminate the overlap occurs in random split method.<br />
<br />
Cross-validation with both splitting methods are provided in the repo to form a comparison. 

