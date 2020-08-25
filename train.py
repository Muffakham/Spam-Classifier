import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA") # setting environment variables for the system, this will store the train.csv location
TEST_DATA = os.environ.get("TEST_DATA")# this will store the test.csv location
FOLD = int(os.environ.get("FOLD"))# this will get us the number of folds defined 
MODEL = os.environ.get("MODEL")# this will get us the model we will use for training

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}
'''
Fold mapping here is a dictionary which will help us in locating the fold which will be used for validation and 
also indicates the folds that will be used for training the data
'''
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True) # selecting the train folds 
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)# selecting the validation fold

    ytrain = train_df.target.values# this gives us all the values in the target column of train dataset
    yvalid = valid_df.target.values# this gives us all the values in the target column of validation dataset 

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)# here the X part of the train dataset is prepared by dropping the columns of outputs
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)# here the X part of the validation dataset is prepared by dropping the columns of outputs

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    '''
    label Encoder's: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
    basically in the code segment below, we are trying to create a label encoding for every column since
    that is the only thing a model would understand
    '''
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())#obtaining the encoding for the specific column in all the datasets 
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    newDataFrame = pd.DataFrame(label_encoders,index=[0])
    newDataFrame.to_csv("input/label_encoded_values.csv", index=True)
    
    # data is ready to train
    clf = dispatcher.MODELS[MODEL] # obtainign the model assigned initially
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1] # this function obtains the prediction as well as the probabilities as well
    print(metrics.roc_auc_score(yvalid, preds))# AUC score for metrics
    filename1 = "models/"+str(MODEL)+'_'+str(FOLD)+'_'+"label_encoder.pkl"
    filename2 = "models/"+str(MODEL)+'_'+str(FOLD)+".pkl"
    filename3 = "models/"+str(MODEL)+'_'+str(FOLD)+'_'+"columns.pkl"
    joblib.dump(label_encoders,filename1 ) #storing the label encoders according to folds as a pickle file 
    joblib.dump(clf,filename2)# storing the trained models according to the fold 
    joblib.dump(train_df.columns,filename3 )# storing the columns as well, used in predict.py