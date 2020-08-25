import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(1):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoderFilename = str(MODEL)+'_'+str(FOLD)+'_'+"label_encoder.pkl"
        columnFilename = str(MODEL)+'_'+str(FOLD)+'_'+"columns.pkl"
        encoders = joblib.load(os.path.join("models",encoderFilename ))
        cols = joblib.load(os.path.join("models",columnFilename ))
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        
        # data is ready to train
        classfierFilename = str(MODEL)+'_'+str(FOLD)+".pkl"
        clf = joblib.load(os.path.join("models",classfierFilename))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub
    

if __name__ == "__main__":
    submission = predict()
    filename = "models/"+str(MODEL)+".csv"
    submission.to_csv(filename, index=False)