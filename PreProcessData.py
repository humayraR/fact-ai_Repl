import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from table_evaluator import TableEvaluator
from ctgan import CTGAN

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import os.path

'''
Cleans the dataset for the CTGAN experiment
'''

def clean_df(direc):
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
    df = pd.read_csv(direc, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

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

    # Create a dataframe to store the synthetic data
    df = df[['race','age','sex','native-country','marital-status','education','occupation','hours-per-week','workclass','relationship','label']]
    return df

discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'label'
]

# Node order contains the order in which to generate the data, starting with the root nodes
node_order = [['race','age','sex','native-country'],['marital-status'],['education'],['occupation','hours-per-week','workclass','relationship'],['label']]
node_order_nl = ['race','age','sex','native-country','marital-status','education','occupation','hours-per-week','workclass','relationship','label']

# List of connections; key is receiving node
node_connections_normal = {'label':['occupation','race','hours-per-week','age','marital-status','education','sex','workclass','native-country','relatinship'],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }

'''
Connections are removed according to the privacy criterion
'''
node_connections_FTU = {'label':['occupation','race','hours-per-week','age','marital-status','education','workclass','native-country','relationship'],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }

node_connections_DP = {'label':['race','age','native-country'],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }

node_connections_CF = {'label':['occupation','race','hours-per-week','age','education','workclass','native-country',],
                    'occupation':['race','age','sex','marital-status','education'],
                    'hours-per-week':['race','age','marital-status','native-country','education','sex'],
                    'workclass':['age','marital-status','sex','education','native-country'],
                    'relationship':['marital-status','education','age','sex','native-country'],
                    'education':['race','age','marital-status','sex','native-country'],
                    'marital-status':['race','age','sex','native-country']
                    }



