## this code produces the ROC curves in the paper

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import tarfile

file = tarfile.open('Data/original_synergy_data.tgz')
# extracting file
file.extractall('./Data')
file.close()
df = pd.read_csv('Data/original_synergy_data.csv')
tissues = df.Tissue.unique().tolist()
print('Tissues', tissues)
folds = 5
cl1 = 'm' #random forest tissue split original
cl2 = 'b' #random forest random split original
cl3 = 'm' #random forest tissue split augmented
cl4 = 'b' #random forest random split augmented
cl5 = 'm' #GBT tissue split original
cl6 = 'b' #GBT random split original
cl7 = 'm' #GBT tissue split augmented
cl8 = 'b' #GBT random split augmented
cl_c = 'k' #chance
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10.5), dpi=200)
####random forest
print("############# RF #######################")
## tissue split
base_fpr = np.linspace(0, 1, 501)
tpr_ts_all = []
aug_tpr_ts_all = []
for ts in tissues:
    tpr = np.load(f'Results/RandomForest_tissue_split/test_tpr_ts_{ts}.npy')
    fpr = np.load(f'Results/RandomForest_tissue_split/test_fpr_ts_{ts}.npy')
    mean_auc = metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=1, alpha=0.5, color=cl1)
    print('original')
    print('ROC %s (AUC = %0.4f)' % (ts, mean_auc))

    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tpr_ts_all.append(tpr)

    tpr_aug = np.load(
        f'Results/RandomForest_tissue_split/aug_test_tpr_ts_{ts}.npy')  # tpr with augmented data
    fpr_aug = np.load(
        f'Results/RandomForest_tissue_split/aug_test_fpr_ts_{ts}.npy')  # fpr with augmented data

    mean_auc_aug = metrics.auc(fpr_aug, tpr_aug)
    ax2.plot(fpr_aug, tpr_aug, lw=1, alpha=0.5, color=cl3)
    print('augmented')
    print('ROC %s (AUC = %0.4f)' % (ts, mean_auc_aug))
    tpr_aug = np.interp(base_fpr, fpr_aug, tpr_aug)
    tpr_aug[0] = 0.0
    aug_tpr_ts_all.append(tpr_aug)

tpr_all = np.asarray(tpr_ts_all)
tpr_all_mean = tpr_all.mean(axis=0)
auc_all_mean = metrics.auc(base_fpr, tpr_all_mean)
ax1.plot(base_fpr, tpr_all_mean, color=cl1,
         label='Tissue-based',
         lw=3, alpha=1)
ax1.set_title('(a)', fontsize=15)
ax1.set_xlabel('FPR', fontsize=15) # X label
ax1.set_ylabel('TPR', fontsize=15) # Y label
ax1.set_aspect('equal', adjustable='box')
print('original Mean ROC (AUC = %0.4f)' % (auc_all_mean))

tpr_all_aug = np.asarray(aug_tpr_ts_all)
aug_tpr_all_mean = tpr_all_aug.mean(axis=0)
aug_auc_all_mean = metrics.auc(base_fpr, aug_tpr_all_mean)
ax2.plot(base_fpr, tpr_all_mean, color=cl3, label='Tissue-based',
         lw=3, alpha=1)
ax2.set_title('(b)', fontsize=15)
ax2.set_xlabel('FPR', fontsize=15) # X label
ax2.set_ylabel('TPR', fontsize=15) # Y label
ax2.set_aspect('equal', adjustable='box')
print('augmented Mean ROC (AUC = %0.4f)' % (aug_auc_all_mean))

## random split
tpr_all_folds = []
aug_tpr_all_folds = []
for fold in range(folds):
    tpr = np.load(f'Results/RandomForest_random_split/test_tpr_fold_{fold}.npy')
    fpr = np.load(f'Results/RandomForest_random_split/test_fpr_fold_{fold}.npy')
    auc = metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=1, alpha=0.5, color=cl2)
    print('original')
    print('ROC %s (AUC = %0.4f)' % (f'fold_{fold}', auc))

    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tpr_all_folds.append(tpr)

    aug_tpr = np.load(f'Results/RandomForest_random_split/aug_test_tpr_fold_{fold}.npy')
    aug_fpr = np.load(f'Results/RandomForest_random_split/aug_test_fpr_fold_{fold}.npy')
    aug_auc = metrics.auc(aug_fpr, aug_tpr)
    ax2.plot(aug_fpr, aug_tpr, lw=1, alpha=0.5, color=cl4)
    print('augmented')
    print('ROC %s (AUC = %0.4f)' % (f'fold_{fold}', aug_auc))
    aug_tpr = np.interp(base_fpr, aug_fpr, aug_tpr)
    aug_tpr[0] = 0.0
    aug_tpr_all_folds.append(aug_tpr)

tprs = np.asarray(tpr_all_folds)
mean_tprs = tprs.mean(axis=0)
mean_auc = metrics.auc(base_fpr, mean_tprs)
ax1.plot(base_fpr, mean_tprs, color=cl2,
         label='Random-split',
         lw=3, alpha=1)
print('original Mean ROC (AUC = %0.4f)' % mean_auc)
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color=cl_c, alpha=.8)
aug_tprs = np.asarray(aug_tpr_all_folds)
aug_mean_tprs = aug_tprs.mean(axis=0)
aug_mean_auc = metrics.auc(base_fpr, aug_mean_tprs)
ax2.plot(base_fpr, aug_mean_tprs, color=cl4,
         label='Random-split',
         lw=3, alpha=1)
print('augmented Mean ROC (AUC = %0.4f)' % aug_mean_auc)
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color=cl_c, alpha=.8)

### Gradient boosting tree
print('################# GBT ######################')
## tissue split
tpr_ts_all = []
aug_tpr_ts_all = []
for ts in tissues:
    tpr_all_eps = []
    aug_tpr_all_eps = []
    tpr = np.load(f'Results/GBT_tissue_split/test_tpr_ts_{ts}.npy')
    fpr = np.load(f'Results/GBT_tissue_split/test_fpr_ts_{ts}.npy')
    mean_auc = metrics.auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=1, alpha=0.5, color=cl5)
    print('original')
    print('ROC %s (AUC = %0.4f)' % (ts, mean_auc))

    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tpr_ts_all.append(tpr)

    tpr_aug = np.load(
        f'Results/GBT_tissue_split/aug_test_tpr_ts_{ts}.npy')  # tpr with augmented data
    fpr_aug = np.load(
        f'Results/GBT_tissue_split/aug_test_fpr_ts_{ts}.npy')  # fpr with augmented data

    mean_auc_aug = metrics.auc(fpr_aug, tpr_aug)
    ax4.plot(fpr_aug, tpr_aug, lw=1, alpha=0.5, color=cl7)
    print('augmented')
    print('ROC %s (AUC = %0.4f)' % (ts, mean_auc_aug))
    tpr_aug = np.interp(base_fpr, fpr_aug, tpr_aug)
    tpr_aug[0] = 0.0
    aug_tpr_ts_all.append(tpr_aug)

tpr_all = np.asarray(tpr_ts_all)
tpr_all_mean = tpr_all.mean(axis=0)
auc_all_mean = metrics.auc(base_fpr, tpr_all_mean)
ax3.plot(base_fpr, tpr_all_mean, color=cl5,
         label='Tissue-based',
         lw=3, alpha=1)
ax3.set_title('(c)', fontsize=15)
ax3.set_xlabel('FPR', fontsize=15) # X label
ax3.set_ylabel('TPR', fontsize=15) # Y label
ax3.set_aspect('equal', adjustable='box')
print('original Mean ROC (AUC = %0.4f)' % (auc_all_mean))

tpr_all_aug = np.asarray(aug_tpr_ts_all)
aug_tpr_all_mean = tpr_all_aug.mean(axis=0)
aug_auc_all_mean = metrics.auc(base_fpr, aug_tpr_all_mean)
ax4.plot(base_fpr, tpr_all_mean, color=cl7, label='Tissue-based',
         lw=3, alpha=1)
ax4.set_title('(d)', fontsize=15)
ax4.set_xlabel('FPR', fontsize=15) # X label
ax4.set_ylabel('TPR', fontsize=15) # Y label
ax4.set_aspect('equal', adjustable='box')
print('augmented Mean ROC (AUC = %0.4f)' % (aug_auc_all_mean))

## random split
tpr_all_folds = []
aug_tpr_all_folds = []
for fold in range(folds):
    tpr = np.load(f'Results/GBT_random_split/test_tpr_fold_{fold}.npy')
    fpr = np.load(f'Results/GBT_random_split/test_fpr_fold_{fold}.npy')
    auc = metrics.auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=1, alpha=0.5, color=cl6)
    print('original')
    print('ROC %s (AUC = %0.4f)' % (f'fold_{fold}', auc))

    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tpr_all_folds.append(tpr)

    aug_tpr = np.load(f'Results/GBT_random_split/aug_test_tpr_fold_{fold}.npy')
    aug_fpr = np.load(f'Results/GBT_random_split/aug_test_fpr_fold_{fold}.npy')
    aug_auc = metrics.auc(aug_fpr, aug_tpr)
    ax4.plot(aug_fpr, aug_tpr, lw=1, alpha=0.5, color=cl8)
    print('augmented')
    print('ROC %s (AUC = %0.4f)' % (f'fold_{fold}', aug_auc))
    aug_tpr = np.interp(base_fpr, aug_fpr, aug_tpr)
    aug_tpr[0] = 0.0
    aug_tpr_all_folds.append(aug_tpr)

tprs = np.asarray(tpr_all_folds)
mean_tprs = tprs.mean(axis=0)
mean_auc = metrics.auc(base_fpr, mean_tprs)
ax3.plot(base_fpr, mean_tprs, color=cl6,
         label='Random-split',
         lw=3, alpha=1)
ax3.plot([0, 1], [0, 1], linestyle='--', lw=2, color=cl_c, alpha=.8)
print('original Mean ROC (AUC = %0.4f)' % (mean_auc))

aug_tprs = np.asarray(aug_tpr_all_folds)
aug_mean_tprs = aug_tprs.mean(axis=0)
aug_mean_auc = metrics.auc(base_fpr, aug_mean_tprs)
ax4.plot(base_fpr, aug_mean_tprs, color=cl8,
         label='Random-split',
         lw=3, alpha=1)
ax4.plot([0, 1], [0, 1], linestyle='--', lw=2, color=cl_c, alpha=.8)
print('augmented Mean ROC (AUC = %0.4f)'% aug_mean_auc)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower right", prop={'size': 15})
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower right", prop={'size': 15})
ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower right", prop={'size': 15})
ax4.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="lower right", prop={'size': 15})

plt.savefig('Results/Figure/ROC_curves.png')
plt.show()
