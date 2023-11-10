import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functions import load_relevant_data_subset
import torch
import torch.nn as nn

import pickle

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import librosa
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/ryu91/kaggle/Google_ISLR_ASL/asl-signs/train.csv")
ROWS_PER_FRAME = 543
def load2(path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet("C:/Users/ryu91/kaggle/Google_ISLR_ASL/asl-signs/"+path.iloc[0].path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
data = load2(data)
print(data.shape)

# ランドマークデータの形状: (50, 100, 3)
#landmark_data = np.random.rand(50, 100, 3)
landmark_data = data[:,200:300,:]

# スペクトログラムの計算
spectrograms = []
for i in range(landmark_data.shape[1]):  # 特徴点の数
    # 1次元のデータを取り出す
    data = landmark_data[:, i, 0]

    # STFTを計算する
    stft = librosa.stft(data.T)

    # STFTの結果をスペクトログラムに変換する
    spectrogram = np.abs(stft)

    # スペクトログラムをリストに追加する
    spectrograms.append(spectrogram)

# スペクトログラムのリストを結合して特徴行列にする
features = np.concatenate(spectrograms, axis=1)
print(features.shape)

# スペクトログラムをプロットする
plt.imshow(features, origin='lower', aspect='auto', cmap='jet')

# 軸の設定
plt.xlabel('Landmark Index')
plt.xlim(0,100)
plt.ylim(0,100)
plt.ylabel('Frame')
plt.colorbar(label='Amplitude')

plt.show()




"""
# pickleファイルを開く
with open('data.pickle', mode='rb') as f:
    data = pickle.load(f)

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
# リストから正解のラベルと予測値のラベルを取り出す
y_true = np.concatenate([np.array(x[0].cpu().numpy()) for x in data])
y_pred = np.concatenate([np.argmax(x[1].cpu().numpy(), axis=1) for x in data])

# 混同行列を計算
cm = confusion_matrix(y_true, y_pred)

# 精度 (Accuracy) を計算
accuracy = accuracy_score(y_true, y_pred)

# 再現率 (Recall) を計算
recall = recall_score(y_true, y_pred, average='macro')

# 適合率 (Precision) を計算
precision = precision_score(y_true, y_pred, average='macro')

# F1 スコアを計算
f1 = f1_score(y_true, y_pred, average='macro')

print(y_true.shape, y_pred.shape)
print(accuracy, recall, precision, f1)
# y_true が正解ラベル、y_pred が予測ラベル
repo = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(repo).transpose().sort_values(by='f1-score', ascending=False)
print(df)
print(df.index[:30])
print(df.index[-50:])
"""


"""# ダミーの入力データを作成
batch_size = 32
frames = 10
channels = 3
landmark = 21
dummy_input = torch.randn(batch_size, frames, landmark, channels)
import torch
import torch.nn as nn

class Conv1DGRUModel(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, num_layers=2, output_size=250, dropout=0.2):
        super(Conv1DGRUModel, self).__init__()

        self.conv1 = nn.Conv3d(input_dim, hidden_size, kernel_size=(3,3,3), padding=3)
        self.bn1 = nn.BatchNorm3d(hidden_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(hidden_size, hidden_size*2, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_size*2)

        self.gru = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size*4, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*4, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, frames, landmarks, 3)
        x = x.permute(0, 3, 2, 1).contiguous()  # (batch_size, landmarks, 3, frames)
        x = x.unsqueeze(-1)  # (batch_size, input_dim=1, landmarks, 3, frames)
        print(x.shape)
        x = self.conv1(x)  # (batch_size, hidden_size, landmarks, 3, frames)
        x = self.bn1(x)
        x = self.relu(x)
        print("conv1:",x.shape)
        x = self.conv2(x)  # (batch_size, hidden_size*2, landmarks, 3, frames)
        x = self.bn2(x)
        x = self.relu(x)
        print(x.shape)

        x = x.permute(0, 4, 2, 3, 1)  # (batch_size, frames, landmarks, 3, hidden_size*2)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch_size, frames, landmarks*3*hidden_size*2)

        x, h = self.gru(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # use the last output only
        return x

model = Conv1DGRUModel()
# モデルにダミーデータを入力
output = model(dummy_input)

# 出力を表示
print(output, output.shape)"""

"""
def gomi(x):
    x = torch.tensor(x)
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

train = pd.read_csv('C:/Users/ryu91/kaggle/Google_ISLR_ASL/asl-signs/train.csv')
print(train.shape)
a = train.iloc[0].path
b = load_relevant_data_subset("C:/Users/ryu91/kaggle/Google_ISLR_ASL/asl-signs/"+a)
print(a)
print(b.shape)
c = gomi(b)
print(c.shape)
print(c)
"""


"""
#left = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/feature_data_left.npy')
right = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/feature_data_right.npy')
label = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/feature_labels.npy')
#print(left, left.shape)
print(right, right.shape)
print(label.shape)
nonzero = np.all(right!=0, axis=1)
print(label[np.all(right!=0, axis=1)], label[np.all(right!=0, axis=1)].shape)

print(right[np.all(right!=0, axis=1), :])

print(right[np.all(right!=0, axis=1), :].shape)
"""

print("all done")