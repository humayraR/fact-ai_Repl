import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def get_metrics(mode,df,synthetic):

    # Split the data into train,test
    traindf, testdf = train_test_split(df, test_size=0.3)
    X_train = traindf.loc[:, traindf.columns != 'label']
    y_train = traindf['label']
    X_test = testdf.loc[:, testdf.columns != 'label']
    y_test = testdf['label']

    clf_df = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                     learning_rate='constant', learning_rate_init=0.001).fit(X_train, y_train)
    '''
    SYNTHETIC DATASET
    '''
    # Make sure the data is representative of the original dataset
    synthetic_balanced_1 = synthetic[synthetic.label == 1].sample(frac = 0.75)
    synthetic_balanced_0 = synthetic[synthetic.label == 0].sample(frac = 0.25)
    #synthetic_balanced = synthetic_balanced_1.append(synthetic_balanced_0)
    synthetic_balanced = pd.concat([synthetic_balanced_1, synthetic_balanced_0], ignore_index=True)

    # Split the data into train,test
    X_syn = synthetic_balanced.loc[:, synthetic_balanced.columns != 'label']
    y_syn = synthetic_balanced['label']

    y_pred_syn = clf_df.predict(X_syn)

    synthetic_pos = synthetic.assign(sex=0)
    synthetic_neg = synthetic.assign(sex=1)
    
    x_pos_syn = synthetic_balanced[synthetic_balanced['sex'] == 0].drop(['label'], axis = 1)[:7508]
    x_neg_syn = synthetic_balanced[synthetic_balanced['sex'] == 1].drop(['label'], axis = 1)[:7508]
    
    pos = clf_df.predict(synthetic_pos.drop('label',axis=1))
    neg = clf_df.predict(synthetic_neg.drop('label',axis=1))

    pred_pos_syn = clf_df.predict(x_pos_syn)
    pred_neg_syn = clf_df.predict(x_neg_syn)
    
    FTU = np.abs(np.mean(pos-neg))
    DP = np.mean(pred_pos_syn)-np.mean(pred_neg_syn)
    
    # Print the obtained statistics
    print('Statistics for dataset for mode:',mode)
    print('Precision:',precision_score(y_syn, y_pred_syn, average='binary'))
    print('Recall:',recall_score(y_syn, y_pred_syn, average='binary'))
    print('AUROC:',roc_auc_score(y_syn, y_pred_syn))
    print('FTU:',FTU)
    print('DP:',DP)