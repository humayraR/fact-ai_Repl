from typing import Any, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(
        G: Any,
        base_mean: float = 0,
        base_var: float = 0.3,
        mean: float = 0,
        var: float = 1,
        SIZE: int = 10000,
        err_type: str = "normal",
        perturb: list = [],
        sigmoid: bool = True,
        expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


def load_adult() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = pd.read_csv(path, names=names, index_col=False)
    # print(df.head(5))
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)
    # print(df.head(5))

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    # print(df.head(5))

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    df_values = df.values
    X = df_values[:, :14].astype(np.uint32)
    y = df_values[:, 14].astype(np.uint8)

    return X, y, df


def get_metrics(mode, df, X_synth, y_synth):
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
    # # Make sure the data is representative of the original dataset
    # synthetic_balanced_1 = synthetic[synthetic.label == 1].sample(22654)
    # synthetic_balanced_0 = synthetic[synthetic.label == 0].sample(7508)
    # synthetic_balanced = synthetic_balanced_1.append(synthetic_balanced_0)

    # Split the data into train,test
    # X_syn = synthetic_balanced.loc[:, synthetic_balanced.columns != 'label']
    # y_syn = synthetic_balanced['label']

    y_pred_syn = clf_df.predict(X_synth)

    synthetic_pos = X_synth.assign(sex=0)
    synthetic_neg = X_synth.assign(sex=1)

    x_pos_syn = X_synth[round(X_synth['sex']) == 0][:7508]
    x_neg_syn = X_synth[round(X_synth['sex']) == 1][:7508]

    pos = clf_df.predict(synthetic_pos)
    neg = clf_df.predict(synthetic_neg)

    pred_pos_syn = clf_df.predict(x_pos_syn)
    pred_neg_syn = clf_df.predict(x_neg_syn)

    FTU = np.abs(np.mean(pos - neg))
    DP = np.mean(pred_pos_syn) - np.mean(pred_neg_syn)

    # Print the obtained statistics
    print('Statistics for dataset for mode:', mode)
    print('Precision:', precision_score(y_synth, y_pred_syn, average='binary'))
    print('Recall:', recall_score(y_synth, y_pred_syn, average='binary'))
    print('AUROC:', roc_auc_score(y_synth, y_pred_syn))
    print('FTU:', FTU)
    print('DP:', DP)

    return df, X_synth, y_synth
