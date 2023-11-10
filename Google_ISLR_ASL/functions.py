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
import multiprocessing as mp
import warnings
warnings.filterwarnings(action='ignore')

# 使用する乱数のseedを固定
import random

BASE_URL = 'C:/Users/ryu91/kaggle/Google_ISLR_ASL'
LANDMARK_FILES_DIR = f"{BASE_URL}/asl-signs/train_landmark_files"
TRAIN_FILE = f"{BASE_URL}/asl-signs/train.csv"
label_map = json.load(open(f"{BASE_URL}/asl-signs/sign_to_prediction_index_map.json", "r"))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def myshape(*args ,**kwargs):
    dis = []
    for i in args:
        try:
            dis.append(i.shape)
        except:
            dis.append(len(i))
    print(dis)

class ALLTime(nn.Module):
    def __init__(self):
        super(ALLTime, self).__init__()
    
    def forward(self, x):
        
        LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        lip_x = x[:, LIP, :].contiguous().view(-1, 40*3)
        #face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        #pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*3)
        #lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        #righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        
        xfeat = torch.cat([lip_x, lefth_x, righth_x], axis=1)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)
        return xfeat
    
#feature_converter=ALLTime()

class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*3)
        
        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        
        x1m = torch.nanmean(face_x, 0)
        x2m = torch.nanmean(lefth_x, 0)
        x3m = torch.nanmean(pose_x, 0)
        x4m = torch.nanmean(righth_x, 0)
        
        x1s = torch.std(face_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)
        
        xfeat = torch.cat([x1m,x2m,x3m,x4m, x1s,x2s,x3s,x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)
        
        return xfeat
    
#feature_converter=FeatureGen()

class myFeatureGen(nn.Module):
    def __init__(self):
        super(myFeatureGen, self).__init__()
        self.face = True
        self.pose = False
        self.right = False
        self.right = False
        pass
    
    def forward(self, x, face=True, pose=True, left = True, right = True, lip=True):
        
        self.face = face
        self.pose = pose
        self.left = left
        self.right = right
        self.lip = lip

        if self.lip:
            LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            ]
            lip_x = x[:, LIP, :].contiguous().view(-1, 40*3)
            x0m = torch.nanmean(lip_x, 0)
            x0s = torch.std(lip_x, 0)
            xlip = torch.cat([x0m, x0s], axis=0)
            xlip = torch.where(torch.isnan(xlip), torch.tensor(0.0, dtype=torch.float32), xlip)
            return xlip
        
        if self.face:
            face_x = x[:,:468,:].contiguous().view(-1, 468*3)
            x1m = torch.nanmean(face_x, 0)
            x1s = torch.std(face_x, 0)
            xface = torch.cat([x1m, x1s], axis=0)
            xface = torch.where(torch.isnan(xface), torch.tensor(0.0, dtype=torch.float32), xface)
            return xface
        
        if self.pose:
            pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
            x3m = torch.nanmean(pose_x, 0)
            x3s = torch.std(pose_x, 0)
            xpose = torch.cat([x3m, x3s], axis=0)
            xpose = torch.where(torch.isnan(xpose), torch.tensor(0.0, dtype=torch.float32), xpose)
            return xpose
        
        if self.left: 
            lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
            lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
            x2m = torch.nanmean(lefth_x, 0)
            x2s = torch.std(lefth_x, 0)
            xleft = torch.cat([x2m, x2s], axis=0)
            xleft = torch.where(torch.isnan(xleft), torch.tensor(0.0, dtype=torch.float32), xleft)
            return xleft
        
        if self.right:
            righth_x = x[:,522:,:].contiguous().view(-1, 21*3)
            righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
            x4m = torch.nanmean(righth_x, 0)
            x4s = torch.std(righth_x, 0)
            xright = torch.cat([x4m, x4s], axis=0)
            xright = torch.where(torch.isnan(xright), torch.tensor(0.0, dtype=torch.float32), xright)
            return xright

#feature_converter=myFeatureGen()

def pre_process(xyz):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common mean
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
    lip = xyz[:, LIP]
    lhand = xyz[:, LHAND]
    rhand = xyz[:, RHAND]
    xyz = torch.cat([ #(none, 82, 3)
        lip,
        lhand,
        rhand,
    ],1)
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_length]
    return xyz

class myFeatureGen(nn.Module):
    def __init__(self):
        super(myFeatureGen, self).__init__()
        self.face = True
        self.pose = False
        self.left = False
        self.right = False
        pass
    
    def forward(self, x, face=True, pose=True, left = True, right = True, lip=False):
        
        self.face = face
        self.pose = pose
        self.left = left
        self.right = right
        self.lip = lip
        if self.lip:
            LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            ]
            lip_x = x[:, LIP, :].contiguous().view(-1, 40*3)
            shape0 = lip_x.shape[0]
            squere = shape0//4
            if shape0 != 0:
                lip_x1 = lip_x[:squere]
                lip_x2 = lip_x[squere: squere*2]
                lip_x3 = lip_x[squere*2: squere*3]
                lip_x4 = lip_x[squere*3:]
                x1m = torch.nanmean(lip_x1, 0)
                x1s = torch.std(lip_x1, 0)
                x2m = torch.nanmean(lip_x2, 0)
                x2s = torch.std(lip_x2, 0)
                x3m = torch.nanmean(lip_x3, 0)
                x3s = torch.std(lip_x3, 0)
                x4m = torch.nanmean(lip_x4, 0)
                x4s = torch.std(lip_x4, 0)
                xlip = torch.cat([x1m, x1s, x2m, x2s, x3m, x3s, x4m, x4s], axis=0)
            elif shape0 == 0:
                x1m = torch.nanmean(lip_x, 0)
                x1s = torch.std(lip_x, 0)
                xlip = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")
            xlip = torch.where(torch.isnan(xlip), torch.tensor(0.0, dtype=torch.float32), xlip)
            return xlip
        if self.pose: 
            pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
            pose_x = pose_x[~torch.any(torch.isnan(pose_x), dim=1),:]
            shape0 = pose_x.shape[0]
            squere = shape0//4
            if shape0 != 0:
                pose_x1 = pose_x[:squere]
                pose_x2 = pose_x[squere: squere*2]
                pose_x3 = pose_x[squere*2: squere*3]
                pose_x4 = pose_x[squere*3:]
                x1m = torch.nanmean(pose_x1, 0)
                x1s = torch.std(pose_x1, 0)
                x2m = torch.nanmean(pose_x2, 0)
                x2s = torch.std(pose_x2, 0)
                x3m = torch.nanmean(pose_x3, 0)
                x3s = torch.std(pose_x3, 0)
                x4m = torch.nanmean(pose_x4, 0)
                x4s = torch.std(pose_x4, 0)
                xpose = torch.cat([x1m, x1s, x2m, x2s, x3m, x3s, x4m, x4s], axis=0)
            elif shape0 == 0:
                x1m = torch.nanmean(pose_x, 0)
                x1s = torch.std(pose_x, 0)
                xpose = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")
            xpose = torch.where(torch.isnan(xpose), torch.tensor(0.0, dtype=torch.float32), xpose)
            return xpose

        if self.left: 
            lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
            lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
            shape0 = lefth_x.shape[0]
            squere = shape0//4
            if shape0 != 0:
                lefth_x1 = lefth_x[:squere]
                lefth_x2 = lefth_x[squere: squere*2]
                lefth_x3 = lefth_x[squere*2: squere*3]
                lefth_x4 = lefth_x[squere*3:]
                x1m = torch.nanmean(lefth_x1, 0)
                x1s = torch.std(lefth_x1, 0)
                x2m = torch.nanmean(lefth_x2, 0)
                x2s = torch.std(lefth_x2, 0)
                x3m = torch.nanmean(lefth_x3, 0)
                x3s = torch.std(lefth_x3, 0)
                x4m = torch.nanmean(lefth_x4, 0)
                x4s = torch.std(lefth_x4, 0)
                xleft = torch.cat([x1m, x1s, x2m, x2s, x3m, x3s, x4m, x4s], axis=0)
            elif shape0 == 0:
                x1m = torch.nanmean(lefth_x, 0)
                x1s = torch.std(lefth_x, 0)
                xleft = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")
            
            xleft = torch.where(torch.isnan(xleft), torch.tensor(0.0, dtype=torch.float32), xleft)
            return xleft
        
        if self.right:
            righth_x = x[:,522:,:].contiguous().view(-1, 21*3)
            righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
            shape0 = righth_x.shape[0]
            squere = shape0//4
            if shape0 != 0:
                righth_x1 = righth_x[:squere]
                righth_x2 = righth_x[squere: squere*2]
                righth_x3 = righth_x[squere*2: squere*3]
                righth_x4 = righth_x[squere*3:]
                x1m = torch.nanmean(righth_x1, 0)
                x1s = torch.std(righth_x1, 0)
                x2m = torch.nanmean(righth_x2, 0)
                x2s = torch.std(righth_x2, 0)
                x3m = torch.nanmean(righth_x3, 0)
                x3s = torch.std(righth_x3, 0)
                x4m = torch.nanmean(righth_x4, 0)
                x4s = torch.std(righth_x4, 0)
                xright = torch.cat([x1m, x1s, x2m, x2s, x3m, x3s, x4m, x4s], axis=0)
            elif shape0 == 0:
                x1m = torch.nanmean(righth_x, 0)
                x1s = torch.std(righth_x, 0)
                xright = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")
            xright = torch.where(torch.isnan(xright), torch.tensor(0.0, dtype=torch.float32), xright)
            return xright

#feature_converter=myFeatureGen()


class TimeFeatureGen(nn.Module):
    def __init__(self):
        super(TimeFeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        #face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        #pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*3)    
        #lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        #righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        lip_x = x[:, LIP, :].contiguous().view(-1, 40*3)
        """
        minface_x = face_x[0,:]
        maxface_x = face_x[-1, :]
        if face_x.shape[0] % 2 == 0:  #2で割り切れるとき
            midface_x = face_x[int(face_x.shape[0] / 2), :]
        else:
            midface_x = face_x[int(face_x.shape[0] // 2), :]

        minpose_x = pose_x[0,:]
        maxpose_x = pose_x[-1, :]
        if pose_x.shape[0] % 2 == 0:  #2で割り切れるとき
            midpose_x = pose_x[int(pose_x.shape[0] / 2), :]
        else:
            midpose_x = pose_x[int(pose_x.shape[0] // 2), :]
        """
        if lip_x.shape[0] != 0:
            minlip_x = lip_x[0,:]
            maxlip_x = lip_x[-1, :]
            if lip_x.shape[0] % 2 == 0:  #2で割り切れるとき
                midlip_x = lip_x[int(lip_x.shape[0] / 2), :]
            else:
                midlip_x = lip_x[int(lip_x.shape[0] // 2), :]

        if lefth_x.shape[0] != 0:
            minlefth_x = lefth_x[0,:]
            maxlefth_x = lefth_x[-1, :]
            if lefth_x.shape[0] % 2 == 0:  #2で割り切れるとき
                midlefth_x = lefth_x[int(lefth_x.shape[0] / 2), :]
            else:
                midlefth_x = lefth_x[int(lefth_x.shape[0] // 2), :]

        if righth_x.shape[0] != 0:
            minrighth_x = righth_x[0,:]
            maxrighth_x = righth_x[-1, :]
            if righth_x.shape[0] % 2 == 0:  #2で割り切れるとき
                midrighth_x = righth_x[int(righth_x.shape[0] / 2), :]
            else:
                midrighth_x = righth_x[int(righth_x.shape[0] // 2), :]

        x = torch.cat([minlip_x, midlip_x, maxlip_x, minlefth_x, midlefth_x, maxlefth_x, minrighth_x, midrighth_x, maxrighth_x], axis=0)
        x = torch.where(torch.isnan(x), torch.tensor(0.0, dtype=torch.float32), x)

        return x
        
    
#time_feature_converter = TimeFeatureGen()

def nanstd(x, dim=0):
    d = x - torch.nanmean(x, dim=dim)
    return torch.sqrt(torch.nanmean(d * d, dim=dim))


class difFeatureGen(nn.Module):
    def __init__(self):
        super(difFeatureGen, self).__init__()
        self.face = True
        self.pose = False
        self.left = False
        self.right = False
        pass
    
    def forward(self, x, face=True, pose=True, left = True, right = True, lip=False):
        
        self.face = face
        self.pose = pose
        self.left = left
        self.right = right
        self.lip = lip

        if self.lip:
            LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            ]
            """
            lip_x = x[:, LIP, :].contiguous().view(-1, 40*3)
            x0m = torch.nanmean(lip_x, 0)
            x0s = torch.std(lip_x, 0)
            xlip = torch.cat([x0m, x0s], axis=0)
            xlip = torch.where(torch.isnan(xlip), torch.tensor(0.0, dtype=torch.float32), xlip)
            """

            #lx = x[:,LIP,0]
            #ly = x[:,LIP,1]
            #lz = x[:,LIP,2]
            lip = x[:, LIP, :].contiguous().view(-1, 40*3)
            shape0 = lip.shape[0]
            devide = 5
            squere = shape0 // devide
            if shape0 >= devide:
                lip_list = []
                for i in range(devide):
                    lipi = lip[squere*i:squere*(i+1), :]
                    #lip2 = lip[squere*(i+1): squere*(i+2), :]
                    lipim = torch.nanmean(lipi, 0)
                    lipis = nanstd(lipi, 0)
                    #lip1m, lip2m = torch.nanmean(lip1, 0), torch.nanmean(lip2, 0)
                    #lip21m = lip2m-lip1m
                    lip_list.append(lipim)
                    lip_list.append(lipis)

                #print(lx_list[0].shape, lx_list[-1].shape)

                lxyzm = torch.cat(lip_list, axis=0)

            elif shape0 < devide:
                #lxm = torch.nanmean(lip, 0)  #size:21
                #print(lxm.shape)
                #x1s = torch.std(lx, 0)
                lxyzm = torch.cat([lip, lip, lip, lip, lip], axis=0)  #一番小さいのが2なので5個
                lxyzm = lxyzm[:10, :].contiguous().view(120*10)

                #xlip = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")

            lxyzm = torch.where(torch.isnan(lxyzm), torch.tensor(0.0, dtype=torch.float32), lxyzm)
            return lxyzm


        if self.face:
            face_x = x[:,:468,:].contiguous().view(-1, 468*3)
            x1m = torch.nanmean(face_x, 0)
            x1s = torch.std(face_x, 0)
            xface = torch.cat([x1m, x1s], axis=0)
            xface = torch.where(torch.isnan(xface), torch.tensor(0.0, dtype=torch.float32), xface)
            return xface
        
        if self.pose:
            pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
            x3m = torch.nanmean(pose_x, 0)
            x3s = torch.std(pose_x, 0)
            xpose = torch.cat([x3m, x3s], axis=0)
            xpose = torch.where(torch.isnan(xpose), torch.tensor(0.0, dtype=torch.float32), xpose)
            return xpose
        
        if self.left:
            
            lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
            #lx = x[:,468:489,0]
            #ly = x[:,468:489,1]
            #lz = x[:,468:489,2]
            shape0 = lefth_x.shape[0]
            devide = 5
            squere = shape0 // devide
            if shape0 >= devide:
                lefth_x_list = []
                for i in range(devide):
                    lefth_xi = lefth_x[squere*i:squere*(i+1), :]
                    #lefth_x2 = lefth_x[squere*(i+1): squere*(i+2), :]
                    lefth_xim = torch.nanmean(lefth_xi, 0)
                    lefth_xis = nanstd(lefth_xi, 0)
                    #lefth_x1m, lefth_x2m = torch.nanmean(lefth_x1, 0), torch.nanmean(lefth_x2, 0)
                    #lefth_x21m = lefth_x2m-lefth_x1m
                    lefth_x_list.append(lefth_xim)
                    lefth_x_list.append(lefth_xis)

                #print(lx_list[0].shape, lx_list[-1].shape)

                lxyzm = torch.cat(lefth_x_list, axis=0)
                

            elif shape0 < devide:
                #lxm = torch.nanmean(lefth_x, 0)  #size:21
                #print(lxm.shape)
                #x1s = torch.std(lx, 0)
                lxyzm = torch.cat([lefth_x, lefth_x, lefth_x, lefth_x, lefth_x], axis=0)  #一番小さいのが2なので5個
                lxyzm = lxyzm[:10, :].contiguous().view(10*21*3)
                #print("@@@@@",lxyzm.shape)

                #xlefth_x = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")

            lxyzm = torch.where(torch.isnan(lxyzm), torch.tensor(0.0, dtype=torch.float32), lxyzm)
            return lxyzm
        if self.right:
            """
            lx = x[:,522:,0]
            ly = x[:,522:,1]
            lz = x[:,522:,2]           
            righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
            x4m = torch.nanmean(righth_x, 0)
            x4s = torch.std(righth_x, 0)
            xright = torch.cat([x4m, x4s], axis=0)
            xright = torch.where(torch.isnan(xright), torch.tensor(0.0, dtype=torch.float32), xright)
            """
            righth_x = x[:,522:,:].contiguous().view(-1, 21*3)

            shape0 = righth_x.shape[0]
            devide = 5
            squere = shape0 // devide
            if shape0 >= devide:
                righth_x_list = []
                for i in range(devide):
                    righth_xi = righth_x[squere*i:squere*(i+1), :]
                    #righth_x2 = righth_x[squere*(i+1): squere*(i+2), :]
                    righth_xim = torch.nanmean(righth_xi, 0)
                    righth_xis = nanstd(righth_xi, 0)
                    #righth_x1m, righth_x2m = torch.nanmean(righth_x1, 0), torch.nanmean(righth_x2, 0)
                    #righth_x21m = righth_x2m-righth_x1m
                    righth_x_list.append(righth_xim)
                    righth_x_list.append(righth_xis)

                #print(lx_list[0].shape, lx_list[-1].shape)

                lxyzm = torch.cat(righth_x_list, axis=0)

            elif shape0 < devide:
                #lxm = torch.nanmean(righth_x, 0)  #size:21
                #print(lxm.shape)
                #x1s = torch.std(lx, 0)
                lxyzm = torch.cat([righth_x, righth_x, righth_x, righth_x, righth_x], axis=0)  #一番小さいのが2なので5個
                lxyzm = lxyzm[:10, :].contiguous().view(630)

                #xrighth_x = torch.cat([x1m, x1s, x1m, x1s, x1m, x1s, x1m, x1s], axis=0)
            else:
                print("ERROR OCCURED!!!!!!!!!")

            lxyzm = torch.where(torch.isnan(lxyzm), torch.tensor(0.0, dtype=torch.float32), lxyzm)
            return lxyzm
#feature_converter = difFeatureGen()


def Angle(data, mainpoint):
    # 肩、肘、手首の3つのポイントから成るベクトルを計算する
    shoulder_to_elbow = data[:, mainpoint, :] - data[:, mainpoint-1, :]
    elbow_to_wrist = data[:, mainpoint+1, :] - data[:, mainpoint, :]
    # ベクトル間の角度(rad)を計算する
    #angle = np.arccos(np.sum(shoulder_to_elbow * elbow_to_wrist, axis=1, keepdims=True) / (np.linalg.norm(shoulder_to_elbow, axis=1, keepdims=True) * np.linalg.norm(elbow_to_wrist, axis=1, keepdims=True)))
    angle = torch.acos(torch.sum(shoulder_to_elbow * elbow_to_wrist, dim=1) / (torch.norm(shoulder_to_elbow, dim=1) * torch.norm(elbow_to_wrist, dim=1)))
    #angle = torch.acos(torch.sum(shoulder_to_elbow * elbow_to_wrist, dim=1, keepdim=True) / (torch.norm(shoulder_to_elbow, dim=1, keepdim=True) * torch.norm(elbow_to_wrist, dim=1, keepdim=True)))

    # Replace NaN values with 0
    angle[torch.isnan(angle)] = 0

    #0~2piなので0~1にする1
    angle = angle / (2*np.pi)

    # 角度の特徴量を出力する
    #print("@",angle)
    return angle

def normalize_data(x):
    # Replace NaN values with 0
    x[torch.isnan(x)] = 0
    
    # Calculate mean and standard deviation ignoring NaN values
    mean = x[~torch.isnan(x)].mean(0, keepdims=True)
    std = x[~torch.isnan(x)].std(0, keepdims=True)
    
    # Normalize data to have zero mean and unit variance
    x = (x - mean) / std
    
    return x

class AngleFeatureGen(nn.Module):
    def __init__(self):
        super(AngleFeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        x = normalize_data(x)

        # x = x[:max_length]

        #face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        #lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        #pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        #righth_x = x[:,522:,:].contiguous().view(-1, 21*3)    

        #lefth
        root_finger1 = Angle(x, 468+2)
        root_finger2 = Angle(x, 468+5)
        root_finger3 = Angle(x, 468+9)
        root_finger4 = Angle(x, 468+13)
        root_finger5 = Angle(x, 468+17)
        joint1_finger1 = Angle(x, 468+3)
        joint1_finger2 = Angle(x, 468+6)
        joint1_finger3 = Angle(x, 468+10)
        joint1_finger4 = Angle(x, 468+14)
        joint1_finger5 = Angle(x, 468+18)
        joint2_finger2 = Angle(x, 468+7)
        joint2_finger3 = Angle(x, 468+11)
        joint2_finger4 = Angle(x, 468+15)
        joint2_finger5 = Angle(x, 468+19)
        print("@",root_finger1.shape, root_finger2.shape, root_finger3.shape, root_finger4.shape, root_finger5.shape)
        root_fingers = torch.cat([root_finger1, root_finger2,root_finger3, root_finger4, root_finger5], axis=0)
        joint1_fingers = torch.cat([joint1_finger1, joint1_finger2,joint1_finger3, joint1_finger4, joint1_finger5], axis=0)
        joint2_fingers = torch.cat([joint2_finger2,joint2_finger3, joint2_finger4, joint2_finger5], axis=0)
        
        Left_angles = torch.cat([root_fingers, joint1_fingers, joint2_fingers], axis=0)
        print("#",len(root_fingers), len(joint1_fingers), len(joint2_fingers), Left_angles.shape)

        #right
        root_finger1 = Angle(x, 522+2)
        root_finger2 = Angle(x, 522+5)
        root_finger3 = Angle(x, 522+9)
        root_finger4 = Angle(x, 522+13)
        root_finger5 = Angle(x, 522+17)
        joint1_finger1 = Angle(x, 522+3)
        joint1_finger2 = Angle(x, 522+6)
        joint1_finger3 = Angle(x, 522+10)
        joint1_finger4 = Angle(x, 522+14)
        joint1_finger5 = Angle(x, 522+18)
        joint2_finger2 = Angle(x, 522+7)
        joint2_finger3 = Angle(x, 522+11)
        joint2_finger4 = Angle(x, 522+15)
        joint2_finger5 = Angle(x, 522+19)

        root_fingers = torch.cat([root_finger1, root_finger2,root_finger3, root_finger4, root_finger5], axis=0)
        joint1_fingers = torch.cat([joint1_finger1, joint1_finger2,joint1_finger3, joint1_finger4, joint1_finger5], axis=0)
        joint2_fingers = torch.cat([joint2_finger2,joint2_finger3, joint2_finger4, joint2_finger5], axis=0)
        Right_angles = torch.cat([root_fingers, joint1_fingers, joint2_fingers], axis=0)

        hand_angles = torch.cat([Left_angles, Right_angles], axis=0)
        print(hand_angles.shape)
        return hand_angles
#feature_converter = AngleFeatureGen()


def reduce_nan(test_tensor):
    # NaNを含まないフレームを取り出す
    nan_mask = torch.isnan(test_tensor).any(dim=2).any(dim=1)
    new_tensor = test_tensor[~nan_mask]
    return new_tensor
def get_frame_indices(n_frames, n_splits=10):
    split_size = n_frames // n_splits  # 1区間あたりのフレーム数
    remainder = n_frames % n_splits  # 最後の区間に追加する余り
    splits = [split_size] * (n_splits-1) + [split_size+remainder]  # 各区間のフレーム数
    indices = [0] + list(np.cumsum(splits))  # 区間の始まりのインデックス
    # return [range(indices[i], indices[i+1]) for i in range(n_splits)]
    return [indices[i] for i in range(n_splits)]
def round_up(root, devide):
    if root / devide != root //devide:
        num = root // devide + 1
    else:
        num = root // devide
    return num
class ReduceFrameFeatureGen(nn.Module):
    def __init__(self):
        super(ReduceFrameFeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        #x = normalize_data(x)

        # x = x[:max_length]
        left_x = x[:,468:489,:]
        right_x = x[:,522:,:]
        #face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        #lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        #pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        #righth_x = x[:,522:,:].contiguous().view(-1, 21*3)

        left_x = reduce_nan(left_x)  #remove nan frame
        right_x = reduce_nan(right_x)  #remove nan frame

        left_frame = left_x.shape[0]
        right_frame = right_x.shape[0]


        if right_frame > left_frame:
            x = right_x
            num_frames = right_frame
        elif left_frame > right_frame:
            x = left_x
            #  lefthand 2 righthand
            x[:,:,0] = -x[:,:,0] 
            num_frames = left_frame
        elif (right_frame == 0) & (left_frame == 0):
            x = torch.tensor([0])
            num_frames = 0
            print("Both hand is NaN!!!")
        elif right_frame == left_frame:
            print("right frame == left frame!!!!!", right_frame, left_frame)
            x = right_x
            num_frames = right_frame
        else:
            print("Any!!!")

        split = 10
        if num_frames > split:
            indexes = get_frame_indices(num_frames, split)
            x = x[indexes,: ,:]
        else:
            num = round_up(split, num_frames)
            x = torch.cat([x]*num, dim=0)
            x = x[:10, :, :]

        #フレームを一次元にする
        #x = x.contiguous().view(-1, 21*3)
        return x
feature_converter = ReduceFrameFeatureGen()



def tf_nan_mean(x, axis=0):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis)

def tf_nan_std(x, axis=0):
    d = x - tf_nan_mean(x, axis=axis)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis))

def flatten_means_and_stds(x, axis=0):
    # Get means and stds
    x_mean = tf_nan_mean(x, axis=0)
    x_std  = tf_nan_std(x,  axis=0)

    x_out = tf.concat([x_mean, x_std], axis=0)
    x_out = tf.reshape(x_out, (1, INPUT_SHAPE[1]*2))
    x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
    return x_out

import tensorflow as tf#; print(f"\t\t TENSORFLOW VERSION: {tf.__version__}");
class OntheshoulderGen(tf.keras.layers.Layer):
    """
    x_listは、モデルの入力データを処理して得られる特徴量を、各セグメントごとに保存するためのリストです。以下のような要素から構成されます。

    平均化された特徴量:averaging_setsに指定されたフレーム数分の特徴量を平均化した結果を、av_setごとに計算し、リストに追加します。リストの各要素は、セグメントごとに計算されます。
    特定のランドマークの特徴量:point_landmarksに指定されたランドマークの特徴量を、リストに追加します。リストの各要素は、セグメントごとに計算されます。
    セグメントごとに分割された特徴量:入力データをSEGMENTSで指定された数に分割し、各セグメントの特徴量をリストに追加します。各要素は、平均値や標準偏差などの統計量が計算されているため、flatten_means_and_stds関数を用いてフラット化されます。
    リサイズされた特徴量:NUM_FRAMES x LANDMARKSの形状にリサイズされた特徴量を、リストに追加します。
    元の特徴量:フラット化された元の特徴量を、リストに追加します。
    以上のように、x_listには、セグメントごとに抽出された様々な種類の特徴量が、リストの各要素として格納されます。
    """
    def __init__(self, DROP_Z, NUM_FRAMES,LANDMARKS, averaging_sets, point_landmarks, SEGMENTS):
        super(OntheshoulderGen, self).__init__()
    
    def call(self, x_in):
        if DROP_Z:
            INPUT_SHAPE = (NUM_FRAMES,LANDMARKS*2)
        else:
            INPUT_SHAPE = (NUM_FRAMES,LANDMARKS*3)
        if DROP_Z:
            x_in = x_in[:, :, 0:2]
        x_list = [tf.expand_dims(tf_nan_mean(x_in[:, av_set[0]:av_set[0]+av_set[1], :], axis=1), axis=1) for av_set in averaging_sets]
        x_list.append(tf.gather(x_in, point_landmarks, axis=1))
        x = tf.concat(x_list, 1)

        x_padded = x
        for i in range(SEGMENTS):
            p0 = tf.where( ((tf.shape(x_padded)[0] % SEGMENTS) > 0) & ((i % 2) != 0) , 1, 0)
            p1 = tf.where( ((tf.shape(x_padded)[0] % SEGMENTS) > 0) & ((i % 2) == 0) , 1, 0)
            paddings = [[p0, p1], [0, 0], [0, 0]]
            x_padded = tf.pad(x_padded, paddings, mode="SYMMETRIC")
        x_list = tf.split(x_padded, SEGMENTS)
        x_list = [flatten_means_and_stds(_x, axis=0) for _x in x_list]

        x_list.append(flatten_means_and_stds(x, axis=0))
        
        ## Resize only dimension 0. Resize can't handle nan, so replace nan with that dimension's avg value to reduce impact.
        x = tf.image.resize(tf.where(tf.math.is_finite(x), x, tf_nan_mean(x, axis=0)), [NUM_FRAMES, LANDMARKS])
        x = tf.reshape(x, (1, INPUT_SHAPE[0]*INPUT_SHAPE[1]))
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x_list.append(x)
        x = tf.concat(x_list, axis=1)
        return x
#feature_converter = OntheshoulderGen()
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


import torch
import torch.nn as nn

class FCBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.2):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(input_channels, output_channels)
        self.bn = nn.BatchNorm1d(output_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

class GetModel(nn.Module):
    def __init__(self, n_labels=250, init_fc=1024, n_blocks=2, dropout_1=0.2, dropout_2=0.4, flat_frame_len=3258):
        super(GetModel, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            input_channels = flat_frame_len if i == 0 else init_fc // (2**(i-1))
            output_channels = init_fc // (2**i)
            dropout = dropout_1 if (1+i) != n_blocks else dropout_2
            self.blocks.append(FCBlock(input_channels, output_channels, dropout))

        self.fc_out = nn.Linear(init_fc // (2**(n_blocks-1)), n_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.fc_out(x)
        #x = self.softmax(x)
        return x

class Conv1DModel(nn.Module):
    """
    input_data: x(batch_size, 時系列データの長さ, 各データのベクトル) => x(batch_size, 各データのベクトル, 時系列データの長さ)
    conv1: conv1d(in_channel, out_channel, kernel, stride)
    """
    def __init__(self, input_dim, hidden_dim, num_classes=250):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1,2)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = x.mean(dim=-1)  # Global average pooling
        print(x.shape)
        #x = x.view(x.size(0), -1)
        x = x.permute(0, 2, 1)  # 時系列データを軸にするために転置する
        x = x.view(x.size(0), 1, x.size(1), x.size(2))  # チャンネル数を1にするために1を挿入する
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        print("@@@"*10)
        return x

class Conv2DModel(nn.Module):
    def __init__(self):
        super(Conv2DModel, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12*12*64,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1,12*12*64)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return f.log_softmax(x, dim=1)


class My3DCNN(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim = 256):
        super(My3DCNN, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        #self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        #self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.num = 256*41  #256*1*5*5
        self.fc1 = nn.Linear(in_features=self.num, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=250)

        self.dropout = nn.Dropout(p=0.1)

        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        print(x.shape)
        # x: (batch_size, channels, frames, keypoints)
        x = x.permute(0, 1, 3, 2)  # permute to (batch_size, channels, keypoints, frames)
        x_list = []
        for i in range(3):  #frame数
            xi = x[:, :, :, i]
            xi = xi.unsqueeze(2)  # add a dimension for the 3D convolution: (batch_size, channels, 1, keypoints, frames)
            xi = xi.unsqueeze(-1)
            xi = F.relu(self.conv3d(xi))  # (batch_size, 64, 1, keypoints, frames)
            xi = self.pool1(xi)  # (batch_size, 64, 1, keypoints/2, frames/2)
            #print(xi.shape)  #torch.Size([32, 256, 1, 10, 1])

            #xi = F.relu(self.conv2(xi))  # (batch_size, 128, 1, keypoints/2, frames/2)
            #xi = self.pool2(xi)  # (batch_size, 128, 1, keypoints/4, frames/4)
            #xi = F.relu(self.conv3(xi))  # (batch_size, 256, 1, keypoints/4, frames/4)
            #xi = self.pool3(xi)  # (batch_size, 256, 1, keypoints/8, frames/8)
            x_list.append(xi)
        x = torch.cat(x_list, axis=0)
        print(x.shape, x_list[0].shape)
        x = x.view(-1, self.num)  # flatten the tensor: (batch_size, 256*1*2*2)
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        #x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 10)

        #x = nn.functional.softmax(x, dim=1)
        return x





class Conv3DModel(nn.Module):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.num = 256  #256*10*5  #256*1*5*5
        self.gap = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc1 = nn.Linear(in_features=self.num, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=250)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x: (batch_size, keypoints, frames, channels)
        x = x.permute(0, 3, 1, 2)  # permute to (batch_size, channels, keypoints, frames)

        x = x.unsqueeze(2)  # add a dimension for the 3D convolution: (batch_size, channels, 1, keypoints, frames)
        x = torch.relu(self.conv1(x))  # (batch_size, 64, 1, keypoints, frames)
        x = self.pool1(x)  # (batch_size, 64, 1, keypoints/2, frames/2)
        #print(x.shape)  #torch.Size([32, 256, 1, 10, 1])

        x = torch.relu(self.conv2(x))  # (batch_size, 128, 1, keypoints/2, frames/2)
        x = self.pool2(x)  # (batch_size, 128, 1, keypoints/4, frames/4)
        
        x = torch.relu(self.conv3(x))  # (batch_size, 256, 1, keypoints/4, frames/4)
        x = self.pool3(x)  # (batch_size, 256, 1, keypoints/8, frames/8)
        #print("2:",x.shape)

        x = self.gap(x)
        print("gap:", x.shape)

        #x = x.view(-1, self.num)  # flatten the tensor: (batch_size, 256*1*2*2)
        x = x.reshape(-1, self.num)
        #print("3:",x.shape)
        x = torch.relu(self.fc1(x))  # (batch_size, 512)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 10)

        #x = nn.functional.softmax(x, dim=1)
        return x



class ASLModel(nn.Module):
    def __init__(self, p, INPUT_LENGTH=3258):
        super(ASLModel, self).__init__()
        self.dropout = nn.Dropout(p)
        self.layer0 = nn.Linear(INPUT_LENGTH, 1024)
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.Linear(512, 250)
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    




# RNNモデルの定義
class RNN(nn.Module):
    def __init__(self, input_size=246, hidden_size=512, num_layers=4, output_size = 250):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # xの形状: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # h0の形状: (num_layers, batch_size, hidden_size)
        out, _ = self.rnn(x, h0)
        # outの形状: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        # outの形状: (batch_size, output_size)
        return out
    
class GRU(nn.Module):
    def __init__(self, input_size=63, hidden_size=256, num_layers=2, output_size=250):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch_size, frames, landmarks, 3) => (batch_size, frames, landmarks*3)
        #print("2:",x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #out, _ = self.gru(x, h0)
        x, h = self.gru(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)
        out = self.fc(x[:, -1, :])
        #out = self.softmax(out)
        return out


import torch
import torch.nn as nn

class Conv1DGRUModel(nn.Module):
    """
    conv1dでlandmarksとxyzを特徴抽出してからGRUに入力
    """
    def __init__(self, input_dim=63, hidden_size=256, hidden_size2 = 256, hidden_size3 = 256, num_layers=2, output_size=250, dropout=0.4):
        super(Conv1DGRUModel, self).__init__()

        # Conv3D layers
        self.conv1 = nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

        # GRU layers
        self.gru = nn.GRU(input_size=hidden_size2, hidden_size=hidden_size3, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size3, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, frames, landmark, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)  #(batch_size, frames, landmark*3)
        x = x.transpose(1,2)  #(batch_size, landmark*3, frames)
        #print(x.shape)

        x = self.conv1(x)  # (batch_size, hidden_state, frames)
        #print(x.shape)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        #print("relu1:",x.shape)
        x = self.conv2(x)  # (batch_size, hidden_state*2, frames)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.dropout(x)

        x = x.transpose(1,2)  # ((batch_size, frames, hidden_state)
        #print(x.shape)
        x, h = self.gru(x)
        #print("GRU:",x.shape)
        
        x = self.fc(x[:, -1, :])  # use the last output only
        return x


ROWS_PER_FRAME = 543

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

class ASLData(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay
        
    def __getitem__(self, index):
        return self.datax[index,:], self.datay[index]
        
    def __len__(self):
        return len(self.datay)
    
class ASLDataX(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay
        
    def __getitem__(self, index):
        return [self.datax[index]], self.datay[index]
        
    def __len__(self):
        return len(self.datay)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, './runs/best_model.pt')
        self.val_loss_min = val_loss


class cv_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, cv_fold, patience=7, verbose=False, delta=0, path='./checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.fold = cv_fold
    def __call__(self, val_loss, model, epoch, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, f'./runs/cv/best_model{self.fold}.pt')
        self.val_loss_min = val_loss