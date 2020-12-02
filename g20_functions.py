# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ds_functions as ds
# import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import sklearn.metrics as metrics
# from sklearn.metrics import mean_absolute_error
from statistics import mean
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFpr, SelectFdr, SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold
register_matplotlib_converters()


# receives lists of results and plots for avgs
def plot_avg_evaluation_results(labels: np.ndarray, train_y, prd_train, test_y, prd_test):
    Accuracy_test = []
    Recall_test = []
    Specificity_test = []
    Precision_test = []
    Accuracy_train = []
    Recall_train = []
    Specificity_train = []
    Precision_train = []
    TN_Test = []
    FN_Test = []
    TP_Test = []
    FP_Test = []
    for i in range(len(train_y)):
        trn_y = train_y[i]
        prd_trn = prd_train[i]
        tst_y = test_y[i]
        prd_tst = prd_test[i]
        cnf_mtx_trn = metrics.confusion_matrix(trn_y, prd_trn, labels)
        tn_trn, fp_trn, fn_trn, tp_trn = cnf_mtx_trn.ravel()
        cnf_mtx_tst = metrics.confusion_matrix(tst_y, prd_tst, labels)
        tn_tst, fp_tst, fn_tst, tp_tst = cnf_mtx_tst.ravel()
        Accuracy_test.append((tn_tst + tp_tst) / (tn_tst + tp_tst + fp_tst + fn_tst))
        Recall_test.append(tp_tst / (tp_tst + fn_tst))
        Specificity_test.append(tn_tst / (tn_tst + fp_tst))
        Precision_test.append(tp_tst / (tp_tst + fp_tst))
        Accuracy_train.append((tn_trn + tp_trn) / (tn_trn + tp_trn + fp_trn + fn_trn))
        Recall_train.append(tp_trn / (tp_trn + fn_trn))
        Specificity_train.append(tn_trn / (tn_trn + fp_trn))
        Precision_train.append(tp_trn / (tp_trn + fp_trn))
        TN_Test.append(tn_tst)
        FN_Test.append(fn_tst)
        TP_Test.append(tp_tst)
        FP_Test.append(fp_tst)
    evaluation = {'Accuracy': [mean(Accuracy_train),
                               mean(Accuracy_test)],
                  'Recall': [mean(Recall_train), mean(Recall_test)],
                  'Specificity': [mean(Specificity_train), mean(Specificity_test)],
                  'Precision': [mean(Precision_train), mean(Precision_test)]}

    fig, axs = plt.subplots(1, 2, figsize=(2 * ds.HEIGHT, ds.HEIGHT))
    ds.multiple_bar_chart(['Train', 'Test'], evaluation, ax=axs[0], title="Model's performance over Train and Test sets")
    ds.plot_confusion_matrix(np.array([[mean(TP_Test), mean(FN_Test)],
                                       [mean(FP_Test), mean(TN_Test)]]), labels, ax=axs[1])


# loads up the heart dataset
def loadHeart():
    data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

    # the time column has to do with how the study kept up with the patients, does not make sense to include in the analysis
    data.pop('time')
    return data

# load heart with the generated feature
def loadHeartGFR():
    data = pd.read_csv('data/heart_with_gfr.csv')
    return data

# load heart with the generated feature and some features excluded
def loadHeartGFRselected():
    data = pd.read_csv('data/heart_with_gfr_f_s.csv')
    return data

# loads up the toxic dataset
def loadToxic():
    return pd.read_csv('data/qsar_oral_toxicity.csv', header=None, sep=';')

def loadToxicBool():
    return pd.read_csv('data/toxic_in_bool.csv', header=None, sep=';')


# discretizes dataset with a one-hot encoding (for heart only)
# TODO returns list with 3 discretization types (quantile & kmeans don't seem to work)
def dummify(data):
    nbins = [10, 2, 50, 2,
             10, 2, 25,
             50, 10, 2, 2]
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
    strategies = ['uniform']  # quantile, kmeans
    data_new = [pd.DataFrame()] * len(strategies)

    for i, strategy in enumerate(strategies):
        for f, n in zip(features, nbins):
            enc = KBinsDiscretizer(n_bins=n, encode='onehot-dense', strategy=strategy)
            dummies_array = enc.fit_transform(data[[f]])
            # print('\ndummies_array\n', dummies_array)
            dummies_df = pd.DataFrame(dummies_array,
                                      columns=[f+'_'+str(j) for j in range(n)])

            data_new[i] = data_new[i].join(dummies_df, how='right')
            # print(data_new[i])

    return data_new


# split the data into x & y
def xySplit(data, target='DEATH_EVENT'):
    y: np.ndarray = data.pop(target).values
    x: np.ndarray = data.values
    labels = pd.unique(y)
    data[target] = y
    return x, y, labels


# returns scaled data with two different methods (for heart only)
def scaleData(data):
    # exclude death_event from scaling, or you'll get problems later
    output = {'Original': pd.DataFrame().append(data)}

    death_event = data.pop('DEATH_EVENT').values
    # print(data.describe(include='all'))
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
    norm_data_zscore = pd.DataFrame(transf.transform(data), columns=data.columns)
    norm_data_zscore['DEATH_EVENT'] = death_event
    output['Z-Score'] = pd.DataFrame().append(norm_data_zscore)
    # print(norm_data_zscore.describe(include='all'))

    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
    norm_data_minmax = pd.DataFrame(transf.transform(data), columns=data.columns)
    norm_data_minmax['DEATH_EVENT'] = death_event
    output['MinMax'] = pd.DataFrame().append(norm_data_minmax)
    # print(norm_data_minmax.describe(include='all'))
    return output


# returns base data, and data for the 3 balancing methods
def balanceData(data, dataset="Heart", save_pics=False):
    if dataset == "Heart":
        target = 'DEATH_EVENT'
    elif dataset == "Toxic":
        target = 1024

    target_count = data[target].value_counts()
    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)
    if save_pics:
        plt.savefig('plots/Class balance for '+dataset+' dataset.png')
    plt.show()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1-ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

    RANDOM_STATE = 42  # The answer to the Ultimate Question of Life, the Universe, and Everything

    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

    df_class_min = data[data[target] == min_class]
    df_class_max = data[data[target] != min_class]
    output = {'Original': pd.DataFrame().append(df_class_min).append(df_class_max)}

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]
    output['UnderSample'] = pd.DataFrame().append(df_class_min).append(df_under)

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]
    output['OverSample'] = pd.DataFrame().append(df_class_max).append(df_over)

    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = data.pop(target).values
    X = data.values
    smote_X, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]
    temp = pd.DataFrame(X, columns=data.columns)
    temp[target] = y
    output['SMOTE'] = temp

    plt.figure()
    ds.multiple_bar_chart([target_count.index[ind_min_class], target_count.index[1-ind_min_class]], values,
                          title='Target', xlabel='frequency', ylabel='Class balance')
    if save_pics:
        plt.savefig('plots/Target for '+dataset+' dataset.png')
    plt.show()
    return output


def outlierRemovalData(data, dataset="Heart"):
    output = {'Original': pd.DataFrame().append(data)}
    if dataset == "Heart":
        target = 'DEATH_EVENT'
    elif dataset == "Toxic":
        target = 1024
    columns = data.columns.copy()
    y = data.pop(target).values
    X = data.values
    for outlier_method in OUTLIER_METHODS:
        for cont in [0.01, 0.05, 0.1]:
            X_if, y_if = outlierRemoval(X, y, outlier_method, cont)
            y_if = np.expand_dims(np.array(y_if), axis=1)
            print(X_if.shape)
            print(y_if.shape)
            data_if = pd.DataFrame(data=np.concatenate((X_if, y_if), axis=1), columns=columns)
            output[outlier_method+" "+str(cont)] = data_if
    return output


# data records, variables and type
def dataShapeandTypes(data):
    print(data.shape)
    print(data.dtypes)


OUTLIER_METHODS = ["Isolation Forest", "Elliptic Envelope", "Local Outlier Factor"]
def outlierRemoval(X, Y, method, cont):
    if method == "Isolation Forest":
        iso = IsolationForest(contamination=cont)
        yhat = iso.fit_predict(X)
    elif method == "Elliptic Envelope":
        ee = EllipticEnvelope(contamination=cont)
        yhat = ee.fit_predict(X)
    elif method == "Local Outlier Factor":
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(X)
    mask = yhat != -1

    return X[mask, :], Y[mask]


# Filters - Classification
# chi2
def fs_chi2(data, dataset, show_indexes=False):
    output = {'Original': pd.DataFrame().append(data)}
    if dataset == "Heart":
        target = 'DEATH_EVENT'
        alphas = (0.005, 0.01, 0.05, 0.1)
    elif dataset == "Toxic":
        target = 1024
        alphas = (1e-10, 1e-8, 1e-6, 1e-5, 1e-4)

    columns = data.columns.copy()
    y = data.pop(target).values
    X = data.values
    data[target] = y
    chi, pval = chi2(X, y)

    for alpha in alphas:
        print('\nalpha =', alpha)

        # --- SelectFpr (false positive rate)
        selector = SelectFpr(chi2, alpha=alpha)
        X_new = selector.fit_transform(X, y)
        print('SelectFpr - Number of features:', len(X_new[0]))
        if show_indexes:
            print('Selected indices:', selector.get_support(indices=True))
        # print('New data space:', X_new[0])
        data_new = pd.DataFrame(data=X_new)
        data_new[target] = y
        output["SelectFpr alpha = {}, {} features selected".format(alpha, len(X_new[0]))] = data_new

        # --- SelectFdr (false discovery rate)
        selector = SelectFdr(chi2, alpha=alpha)
        X_new = selector.fit_transform(X, y)
        print('SelectFdr - Number of features:', len(X_new[0]))
        if show_indexes:
            print('Selected indices:', selector.get_support(indices=True))
        # print('New data space:', X_new[0])
        data_new = pd.DataFrame(data=X_new)
        data_new[target] = y
        output["SelectFdr alpha = {}, {} features selected".format(alpha, len(X_new[0]))] = data_new

        # --- SelectFwe (family wise error)
        selector = SelectFwe(chi2, alpha=alpha)
        X_new = selector.fit_transform(X, y)
        print('SelectFwe - Number of features:', len(X_new[0]))
        if show_indexes:
            print('Selected indices:', selector.get_support(indices=True))
        # print('New data space:', X_new[0])
        data_new = pd.DataFrame(data=X_new)
        data_new[target] = y
        output["SelectFwe alpha = {}, {} features selected".format(alpha, len(X_new[0]))] = data_new

    return output


def fs_k_best(data, dataset, show_indexes=False):
    output = {'Original': pd.DataFrame().append(data)}
    if dataset == "Heart":
        target = 'DEATH_EVENT'
        ks = (2, 3, 4, 5, 6, 7, 8, 9)
    elif dataset == "Toxic":
        target = 1024
        ks = (10, 20, 50, 100, 200, 300)

    columns = data.columns.copy()
    y = data.pop(target).values
    X = data.values
    data[target] = y

    for k in ks:
        selector = SelectKBest(mutual_info_classif, k)
        X_new = selector.fit_transform(X, y)
        # print('\nSelectKBest scores:\n', selector.scores_)
        print("\nSelectKBest k = {}, {} features selected".format(k, len(X_new[0])))
        if show_indexes:
            print('\nSelected indices: {}'.format(selector.get_support(indices=True)))
        data_new = pd.DataFrame(data=X_new)
        data_new[target] = y
        output['k={}'.format(k)] = data_new

    return output


def fs_percentile(data, dataset, show_indexes=False):
    output = {'Original': pd.DataFrame().append(data)}
    if dataset == "Heart":
        target = 'DEATH_EVENT'
        percentiles = (20, 30, 40, 50, 60, 70, 80)
    elif dataset == "Toxic":
        target = 1024
        percentiles = (3, 5, 7, 10, 15, 20)

    columns = data.columns.copy()
    y = data.pop(target).values
    X = data.values
    data[target] = y

    for percentile in percentiles:
        selector = SelectPercentile(f_classif, percentile=percentile)
        X_new = selector.fit_transform(X, y)
        # print('\nSelectPercentile p-values:\n', selector.pvalues_)
        print('\nSelectPercentile ({}%) - Number of features: {}'.format(percentile, len(X_new[0])))
        if show_indexes:
            print('Selected indices:', selector.get_support(indices=True))
        # print('New data space:', X_new[0])
        data_new = pd.DataFrame(data=X_new)
        data_new[target] = y
        output["SelectPercentile {}%, {} features selected".format(percentile, len(X_new[0]))] = data_new

    return output


def fs_variance_threshold(data, dataset, show_indexes=False):
    output = {'Original': pd.DataFrame().append(data)}
    if dataset == "Heart":
        target = 'DEATH_EVENT'
        thresholds = (0.01, 0.1, 0.2, 0.3, 0.5)
    elif dataset == "Toxic":
        target = 1024
        thresholds = (0.1, 0.15, 0.2)

    columns = data.columns.copy()
    y = data.pop(target).values
    X = data.values
    data[target] = y

    for threshold in thresholds:
        selector = VarianceThreshold(threshold)
        X_new = selector.fit_transform(X)
        print('Variance (threshold={}) - Number of features: {}'.format(threshold, len(X_new[0])))
        if show_indexes:
            print('Selected indices:', selector.get_support(indices=True))
        # print('New data space:', X_new[0])
        data_new = pd.DataFrame(data=X_new)
        data_new[target] = y
        output["SelectKBest thres = {}, {} features selected".format(threshold, len(X_new[0]))] = data_new

    return output
