## this file run the gradient boosting trees with original and augmented dataset under tissue split

import pandas as pd
from numpy import *
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, f1_score
from collections import Counter
import tarfile


file = tarfile.open('Data/original_synergy_data.tgz')
# extracting file
file.extractall('./Data')
file.close()

df = pd.read_csv('Data/original_synergy_data.csv')
c = dict(Counter(df['class']))
# open file
file = tarfile.open('Data/GBT_tissue_split/breast.tgz')
# extracting file
file.extractall('./Data/GBT_tissue_split')
file.close()

tissues = df.Tissue.unique().tolist()
columns = tissues
df_tr_acc = pd.DataFrame(columns=columns)
df_tr_auc = pd.DataFrame(columns=columns)
df_test_acc = pd.DataFrame(columns=columns)
df_test_auc = pd.DataFrame(columns=columns)

df_aug_tr_acc = pd.DataFrame(columns=columns)
df_aug_tr_auc = pd.DataFrame(columns=columns)
df_aug_test_acc = pd.DataFrame(columns=columns)
df_aug_test_auc = pd.DataFrame(columns=columns)

df_precision_score = pd.DataFrame(columns=columns)
df_recall_score = pd.DataFrame(columns=columns)
df_f1 = pd.DataFrame(columns=columns)

df_precision_score_aug = pd.DataFrame(columns=columns)
df_recall_score_aug = pd.DataFrame(columns=columns)
df_f1_aug = pd.DataFrame(columns=columns)

num_fold = 5
fold_cnt = 0
FPR_cal = np.zeros(num_fold)
MCC_cal = np.zeros(num_fold)
FPR_cal_aug = np.zeros(num_fold)
MCC_cal_aug = np.zeros(num_fold)
for ts in tissues:
    # print(ts)
    test_set = df[df['Tissue'] == ts]
    train_set = df[df['Tissue'] != ts]

    sample_keep = pd.read_csv(f'Data/GBT_tissue_split/{ts}.csv')
    aug_sampled = sample_keep

    train_aug_feature = pd.concat([aug_sampled.iloc[:, 8:908], train_set.iloc[:, 7:907]])
    train_aug_label = pd.concat([aug_sampled.iloc[:, 6], train_set.iloc[:, 5]])
    train_aug_label = train_aug_label.astype('int')

    random.seed(123)

    clf = GradientBoostingClassifier(n_estimators=650, min_samples_leaf=120, max_features='sqrt',
                                          learning_rate=0.28, max_depth=5, random_state=random.seed(123))
    clf.fit(train_set.iloc[:, 7:907], train_set.iloc[:, 5])

    y_train_pred = clf.predict(train_set.iloc[:, 7:907])
    train_acc = metrics.accuracy_score(train_set.iloc[:, 5], y_train_pred)
    y_train_score = clf.predict_proba(train_set.iloc[:, 7:907])
    np.save(f'Results/GBT_tissue_split/y_train_score_{ts}.npy', y_train_score)

    fpr_train, tpr_train, thresholds_train = roc_curve(train_set.iloc[:, 5], y_train_score[:, 1], pos_label=1)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)

    df_tr_acc.at[0, ts] = train_acc
    df_tr_auc.at[0, ts] = roc_auc_train

    y_pred = clf.predict(test_set.iloc[:, 7:907])
    acc = metrics.accuracy_score(test_set.iloc[:, 5], y_pred)
    y_score = clf.predict_proba(test_set.iloc[:, 7:907])
    fpr, tpr, thresholds = roc_curve(test_set.iloc[:, 5], y_score[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # print(roc_auc)
    df_test_acc.at[0, ts] = acc
    df_test_auc.at[0, ts] = roc_auc

    # np.save(f'Results/GBT_tissue_split/train_fpr_fold_{fold_cnt}.npy', fpr_train)
    # np.save(f'Results/GBT_tissue_split/train_tpr_fold_{fold_cnt}.npy', tpr_train)

    np.save(f'Results/GBT_tissue_split/test_fpr_ts_{ts}.npy', fpr)
    np.save(f'Results/GBT_tissue_split/test_tpr_ts_{ts}.npy', tpr)

    confusionmatrix = confusion_matrix(test_set.iloc[:, 5], y_pred)
    p_score = precision_score(test_set.iloc[:, 5], y_pred)
    r_score = recall_score(test_set.iloc[:, 5], y_pred)
    f1 = f1_score(test_set.iloc[:, 5], y_pred)

    random.seed(123)

    clf_aug = GradientBoostingClassifier(n_estimators=650, min_samples_leaf=120, max_features='sqrt',
                                          learning_rate=0.28, max_depth=5, random_state=random.seed(123))
    clf_aug.fit(train_aug_feature, train_aug_label)

    y_train_aug_pred = clf_aug.predict(train_aug_feature)
    train_acc_aug = metrics.accuracy_score(train_aug_label, y_train_aug_pred)
    y_train_aug_score = clf_aug.predict_proba(train_aug_feature)
    # np.save(f'Results/GBT_tissue_split/y_train_score_aug_{ts}.npy', y_train_aug_score)
    fpr_aug_train, tpr_aug_train, thresholds_aug_train = roc_curve(train_aug_label, y_train_aug_score[:, 1],
                                                                   pos_label=1)
    roc_auc_aug_train = metrics.auc(fpr_aug_train, tpr_aug_train)
    # print(roc_auc_aug_train)
    df_aug_tr_acc.at[0, ts] = train_acc_aug
    df_aug_tr_auc.at[0, ts] = roc_auc_aug_train

    y_pred_aug = clf_aug.predict(test_set.iloc[:, 7:907])
    acc_aug = metrics.accuracy_score(test_set.iloc[:, 5], y_pred_aug)
    y_aug_score = clf_aug.predict_proba(test_set.iloc[:, 7:907])
    fpr_aug, tpr_aug, thresholds_aug = roc_curve(test_set.iloc[:, 5], y_aug_score[:, 1], pos_label=1)
    roc_auc_aug = metrics.auc(fpr_aug, tpr_aug)
    # print(roc_auc_aug)
    df_aug_test_acc.at[0, ts] = acc_aug
    df_aug_test_auc.at[0, ts] = roc_auc_aug

    np.save(f'Results/GBT_tissue_split/aug_test_fpr_ts_{ts}.npy', fpr_aug)
    np.save(f'Results/GBT_tissue_split/aug_test_tpr_ts_{ts}.npy', tpr_aug)

    confusionmatrix_aug = confusion_matrix(test_set.iloc[:, 5], y_pred_aug)
    p_score_aug = precision_score(test_set.iloc[:, 5], y_pred_aug)
    r_score_aug = recall_score(test_set.iloc[:, 5], y_pred_aug)
    f1_aug = f1_score(test_set.iloc[:, 5], y_pred_aug)

    TP = confusionmatrix[1][1]
    TN = confusionmatrix[0][0]
    FP = confusionmatrix[0][1]
    FN = confusionmatrix[1][0]
    # print('TP', TP)
    # print('TN', TN)
    # print('FP', FP)
    # print('FN', FN)
    fpr_fromCF = FP / (FP + TN)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    TP_aug = confusionmatrix_aug[1][1]
    TN_aug = confusionmatrix_aug[0][0]
    FP_aug = confusionmatrix_aug[0][1]
    FN_aug = confusionmatrix_aug[1][0]
    # print('TP_aug', TP_aug)
    # print('TN_aug', TN_aug)
    # print('FP_aug', FP_aug)
    # print('FN_aug', FN_aug)
    fpr_aug_fromCF = FP_aug / (FP_aug + TN_aug)
    mcc_aug = (TP_aug * TN_aug - FP_aug * FN_aug) / np.sqrt(
        (TP_aug + FP_aug) * (TP_aug + FN_aug) * (TN_aug + FP_aug) * (TN_aug + FN_aug))

    FPR_cal[fold_cnt] = fpr_fromCF
    MCC_cal[fold_cnt] = mcc
    FPR_cal_aug[fold_cnt] = fpr_aug_fromCF
    MCC_cal_aug[fold_cnt] = mcc_aug

    np.save(f'Results/GBT_tissue_split/confusion_matrix_ts_{ts}.npy', confusionmatrix)
    np.save(f'Results/GBT_tissue_split/confusion_matrix_aug_ts_{ts}.npy', confusionmatrix_aug)

    df_precision_score.at[0, ts] = p_score
    df_precision_score_aug.at[0, ts] = p_score_aug
    df_recall_score.at[0, ts] = r_score
    df_recall_score_aug.at[0, ts] = r_score_aug
    df_f1.at[0, ts] = f1
    df_f1_aug.at[0, ts] = f1_aug

    fold_cnt += 1



mean_fpr = np.mean(FPR_cal)
aug_mean_fpr = np.mean(FPR_cal_aug)
mean_mcc = np.mean(MCC_cal)
aug_mean_mcc = np.mean(MCC_cal_aug)
df_tr_auc['mean'] = df_tr_auc.mean(axis=1)
df_test_auc['mean'] = df_test_auc.mean(axis=1)
df_aug_tr_auc['mean'] = df_aug_tr_auc.mean(axis=1)
df_aug_test_auc['mean'] = df_aug_test_auc.mean(axis=1)

df_tr_acc['mean'] = df_tr_acc.mean(axis=1)
df_test_acc['mean'] = df_test_acc.mean(axis=1)
df_aug_tr_acc['mean'] = df_aug_tr_acc.mean(axis=1)
df_aug_test_acc['mean'] = df_aug_test_acc.mean(axis=1)

df_precision_score['mean'] = df_precision_score.mean(axis=1)
df_precision_score_aug['mean'] = df_precision_score_aug.mean(axis=1)
df_recall_score['mean'] = df_recall_score.mean(axis=1)
df_recall_score_aug['mean'] = df_recall_score_aug.mean(axis=1)
df_f1['mean'] = df_f1.mean(axis=1)
df_f1_aug['mean'] = df_f1_aug.mean(axis=1)

print('accuracy:', df_test_acc['mean'].values)
print('aug_accuracy', df_aug_test_acc['mean'].values)
print('auc:', df_test_auc['mean'].values)
print('aug_auc', df_aug_test_auc['mean'].values)
print('recall:', df_recall_score['mean'].values)
print('aug_recall:',df_recall_score_aug['mean'].values)
print('precision:',df_precision_score['mean'].values)
print('aug_pression:',df_precision_score_aug['mean'].values)
print('f1:',df_f1['mean'].values)
print('aug_f1:',df_f1_aug['mean'].values)
print('MCC:', mean_mcc, 'aug_MCC:', aug_mean_mcc)
print('FPR:', mean_fpr, 'aug_FPR:', aug_mean_fpr)


df_tr_acc.to_csv('Results/GBT_tissue_split/train_accuracy.csv', index=False)
df_tr_auc.to_csv('Results/GBT_tissue_split/train_roc_auc.csv', index=False)
df_test_acc.to_csv('Results/GBT_tissue_split/test_accuracy.csv', index=False)
df_test_auc.to_csv('Results/GBT_tissue_split/test_roc_auc.csv', index=False)

df_aug_tr_acc.to_csv('Results/GBT_tissue_split/aug_train_accuracy.csv', index=False)
df_aug_tr_auc.to_csv('Results/GBT_tissue_split/aug_train_roc_auc.csv', index=False)
df_aug_test_acc.to_csv('Results/GBT_tissue_split/aug_test_accuracy.csv', index=False)
df_aug_test_auc.to_csv('Results/GBT_tissue_split/aug_test_roc_auc.csv', index=False)

df_precision_score.to_csv('Results/GBT_tissue_split/precision_score.csv', index=False)
df_precision_score_aug.to_csv('Results/GBT_tissue_split/recision_score_aug.csv', index=False)
df_recall_score.to_csv('Results/GBT_tissue_split/recall_score.csv', index=False)
df_recall_score_aug.to_csv('Results/GBT_tissue_split/recall_score_aug.csv', index=False)
df_f1.to_csv('Results/GBT_tissue_split/f1_score.csv', index=False)
df_f1_aug.to_csv('Results/GBT_tissue_split/f1_score_aug.csv', index=False)
