import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42) #stratified K fold takes the data and divides it into various folds

    '''
    Here in this case the above function divides the data into five folds
    check this lionk for folds explanation: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    this technique is used for the purpose of cross validation and 
    hence divides the data into training set and validation set respectively
 
    '''


    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        '''
        here the split function divides the data into various folds. 
        each fold is any of the [0,1,2,3,4] becasue we wanted five folds
        so this peice of code divides the data and displays the number of values in both train set and validation set
        '''
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold # A new column called fold is created and contains the number of corresponding folds
    

    df.to_csv("input/train_folds.csv", index=False)