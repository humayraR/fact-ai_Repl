import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from fairGAN import Medgan
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_fairgan(data):
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    #data = np.load(args.data_file, allow_pickle = True)
    inputDim = data.shape[1]-1
    inputNum = data.shape[0]

    mg = Medgan(dataType=args.data_type,
                inputDim=inputDim,
                embeddingDim=args.embed_size,
                randomDim=args.noise_size,
                generatorDims=args.generator_size,
                discriminatorDims=args.discriminator_size,
                compressDims=args.compressor_size,
                decompressDims=args.decompressor_size,
                bnDecay=args.batchnorm_decay,
                l2scale=args.L2)


    mg.train(dataPath=args.data_file,
             modelPath=args.model_file,
             outPath=args.out_file,
             pretrainEpochs=args.n_pretrain_epoch,
             nEpochs=args.n_epoch,
             discriminatorTrainPeriod=args.n_discriminator_update,
             generatorTrainPeriod=args.n_generator_update,
             pretrainBatchSize=args.pretrain_batch_size,
             batchSize=args.batch_size,
             saveMaxKeep=args.save_max_keep)

    return mg.generateData(nSamples=inputNum,
                        modelFile=args.model_file,
                        batchSize=args.batch_size,
                        outFile=args.out_file)

data = pd.read_csv("~/Desktop/causal_dp/data/census_data.csv")
print(len(data))

def label_fix(label):
    if label==' <=50K':
        return 0
    else:
        return 1
    
data['income_bracket'] = data['income_bracket'].apply(label_fix)




gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country,
            age, education_num, capital_gain, capital_loss, hours_per_week]
from sklearn.feature_selection import mutual_info_classif
print("Base MI RACE;INCOME = ", mutual_info_classif(np.expand_dims(data['income_bracket'], -1), data['race'], discrete_features = True))
print("Base MI Gender;INCOME = ", mutual_info_classif(np.expand_dims(data['income_bracket'], -1), data['gender'], discrete_features = True))



from sklearn import preprocessing
for feat in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']:
    data[feat] = preprocessing.LabelEncoder().fit_transform(data[feat])

data.drop(['capital_gain', 'capital_loss', 'education'], axis = 1, inplace = True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data)
data[data.columns] = scaler.fit_transform(data)

print(train_fairgan(data))