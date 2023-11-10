import pandas as pd
import json


BASE_URL = 'C:/Users/ryu91/kaggle/Google_ISLR_ASL'
LANDMARK_FILES_DIR = f"{BASE_URL}/asl-signs/train_landmark_files"
TRAIN_FILE = f"{BASE_URL}/asl-signs/train.csv"
label_map = json.load(open(f"{BASE_URL}/asl-signs/sign_to_prediction_index_map.json", "r"))

train=pd.read_csv(TRAIN_FILE)
all_index = list(range(train.shape[0]))
all_index = set(all_index)
pid = train["participant_id"].unique()
pid.sort()

def myCV(data, num_fold=10,val_fold=1, train = train):
    """
    途中までしかつくってない
    num_fold=10しか考えてない
    """
    split = len(pid)//num_fold#participant_idは21個
    if split != len(pid)/ num_fold:
        BOOL_LAST = True
    else:
        BOOL_LAST = False
    for i in range(num_fold):
        splited = pid[split*i:split*(i+1)]
        train.loc[(train['participant_id'] >= splited.min()) & (train['participant_id'] <= splited.max()), 'fold'] = i
        if (i == num_fold-1) & (BOOL_LAST==True):
            train.loc[train['participant_id'] >= splited.min(), 'fold'] = i
    if val_fold > num_fold:
        print("ERROR!!!!!!!!!!!!!!!!!!!")
    elif val_fold == num_fold-1:
        val_index = set(train[train["fold"]==val_fold].index)
        test_index = set(train[train["fold"]==0].index)
    else:
        val_index = set(train[train["fold"]==val_fold].index)
        test_index = set(train[train["fold"]==val_fold+1].index)
    train_indexs = list(all_index - (val_index | test_index))
    val_index = list(val_index)
    test_index = list(test_index)

    train_indexs.sort()

    train = data[train_indexs]
    val = data[val_index]
    test = data[test_index]
    return train, val, test

def myshape(*args ,**kwargs):
    dis = []
    for i in args:
        dis.append(i.shape)
    print(dis)

