import os
import gc
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')
import multiprocessing as mp
import wandb
from torch.optim.lr_scheduler import OneCycleLR

#import functions from anoter files
from functions import *
from config import CONFIG
from helper_functions import myCV, myshape
from preprocess import *
from train import train, cv_train
from cv_training import training, cv_training
seed_everything()

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.device_count()) # returns 1 in my case

    BASE_URL = 'C:/Users/ryu91/kaggle/Google_ISLR_ASL'
    LANDMARK_FILES_DIR = f"{BASE_URL}/asl-signs/train_landmark_files"
    TRAIN_FILE = f"{BASE_URL}/asl-signs/train.csv"
    label_map = json.load(open(f"{BASE_URL}/asl-signs/sign_to_prediction_index_map.json", "r"))
    def read_json_file(file_path):
        try:
            # Open the file and load the JSON data into a Python object
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            return json_data
        except FileNotFoundError:
            # Raise an error if the file path does not exist
            raise FileNotFoundError(f"File not found: {file_path}")
        except ValueError:
            # Raise an error if the file does not contain valid JSON data
            raise ValueError(f"Invalid JSON data in file: {file_path}")
    json_path = (BASE_URL+ "/asl-signs/sign_to_prediction_index_map.json")
    print("path:",json_path)
    s2p_map = {k.lower():v for k,v in read_json_file(json_path).items()}
    p2s_map = {v:k for k,v in read_json_file(json_path).items()}
    encoder = lambda x: s2p_map.get(x.lower())
    decoder = lambda x: p2s_map.get(x)

    CONFIG = CONFIG()
    do_wandb = CONFIG.do_wandb
    if do_wandb:
        wandb_api = '7f0b0b26236eae67c25381f94363123b539faad3'
        wandb.login(key=wandb_api)
        run = wandb.init(config=CONFIG, project="GoogleISLR_ASL", name=CONFIG.RUN_NAME)  #, config=hyperparameters)
        assert run is wandb.run

    #feature_converter = difFeatureGen()

    ROWS_PER_FRAME = 543


    #こっから
    DROP_Z = False
    NUM_FRAMES = 15
    SEGMENTS = 5
    LEFT_HAND_OFFSET = 468
    POSE_OFFSET = LEFT_HAND_OFFSET+21
    RIGHT_HAND_OFFSET = POSE_OFFSET+33
    ## average over the entire face, and the entire 'pose'
    averaging_sets = [[0, 468], [POSE_OFFSET, 33]]
    lip_landmarks = [61, 185, 40, 39, 37,  0, 267, 269, 270, 409,
                    291,146, 91,181, 84, 17, 314, 405, 321, 375, 
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 
                    95, 88, 178, 87, 14,317, 402, 318, 324, 308]
    left_hand_landmarks = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET+21))
    right_hand_landmarks = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET+21))
    point_landmarks = [item for sublist in [lip_landmarks, left_hand_landmarks, right_hand_landmarks] for item in sublist]
    LANDMARKS = len(point_landmarks) + len(averaging_sets)
    #print(LANDMARKS)
    INPUT_SHAPE = (NUM_FRAMES,LANDMARKS*3)

    #feature_converter = OntheshoulderGen(DROP_Z, NUM_FRAMES, LANDMARKS, averaging_sets, point_landmarks, SEGMENTS)
    #ここまでontheshoulder用

    feature_converter = ReduceFrameFeatureGen()
    #convert_and_save_data(False)
    #myconvert_and_save_data(False)
    #time_convert_and_save_data(False)


    #shoulder_convert_and_save_data(INPUT_SHAPE, SEGMENTS)

    #dataxのフレーム数は2から344の長さ

    #datax = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/all_feature_data.npy', allow_pickle=True)
    #print(len(datax), datax[0].shape, datax[10].shape, datax[-1].shape)
    #datax = np.load(f"{BASE_URL}/feature_datas/time_feature_datas.npy")
    #lip = np.load(f'{BASE_URL}/dif_feature_data_lip.npy', allow_pickle=True)
    #left = np.load(f"{BASE_URL}/dif_feature_data_left.npy", allow_pickle=True)
    #right = np.load(f"{BASE_URL}/dif_feature_data_right.npy", allow_pickle=True)

    #oneleft = np.load(f"{BASE_URL}/feature_datas/feature_data_left.npy")

    #lip = lip.reshape(-1, 5, 240)
    #left = left.reshape(-1, 5, 126)
    #right = right.reshape(-1, 5, 126)
    #print(lip.shape, left.shape, right.shape)
    
    #face = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/feature_data_face.npy')
    #pose = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/feature_data_pose.npy')
    # テンソルの次元を入れ替える

    
    datax = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/reduce_hand/reduce_10feature_datas.npy', allow_pickle=True)
    #datax = datax.reshape(datax.shape[0], -1)  #flatten
    print(datax.shape)

    """
    r1 = right[:,:21,:].transpose(0, 2, 1)
    r2 = right[:,21:21*2, :].transpose(0, 2, 1)
    r3 = right[:,21*2:21*3, :].transpose(0, 2, 1)
    lip1 = lip[:,:40,:].transpose(0, 2, 1)
    lip2 = lip[:,40:40*2, :].transpose(0, 2, 1)
    lip3 = lip[:,40*2:40*3, :].transpose(0, 2, 1)
    """
    # 二次元目を21で10等分
    """
    split_x = np.array_split(left, 21, axis=1)
    leftx = np.stack(split_x, axis=1)
    print(leftx.shape)
    datax = leftx
    #rightx = np.stack([r1, r2, r3], axis=2)
    #lipx = np.stack([lip1, lip2, lip3], axis=2)
    #print(leftx.shape, rightx.shape, lipx.shape)
    # 新しい次元を追加する
    #datax = np.concatenate([leftx, rightx, lipx], axis=-1) # 最後の次元で結合
    #datax = np.concatenate([lip, left, right], axis=1)
    #nonzero = np.all(datax!=0, axis=3)
    #datax = datax[nonzero, :]
    #print(left.shape)
    #print(left.shape, right.shape)
    #datax = np.concatenate([lip, left, right], axis=1)
    print(datax.shape)
    #datax = np.load(f"{BASE_URL}/feature_datas/feature_datas.npy")
    datay = np.load(f"{BASE_URL}/feature_datas/feature_labels.npy")
    #datay = datay[nonzero]
    """
    datay = np.load(f"{BASE_URL}/feature_datas/feature_labels.npy")
    print(type(datay), type(datax))
    print(myshape(datax, datay))

    train_df=pd.read_csv(TRAIN_FILE)
    all_index = list(range(train_df.shape[0]))
    all_index = set(all_index)
    pid = train_df["participant_id"].unique()
    pid.sort()

    EPOCHS = CONFIG.EPOCHS
    BATCH_SIZE = CONFIG.BATCH_SIZE

    #trainx, testx, trainy, testy = train_test_split(datax, datay, test_size=0.15, random_state=42)
    #fold:1,2,3  ex)fold==1:val, fold!=1:train
    training(datax, datay, EPOCHS, BATCH_SIZE, do_wandb, device)


    #read
    #f = open("./list.txt","rb")
    #list_row = pickle.load(f)