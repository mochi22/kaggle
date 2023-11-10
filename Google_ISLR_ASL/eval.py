from main import *
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

import pickle

datax = np.load('C:/Users/ryu91/kaggle/Google_ISLR_ASL/feature_datas/reduce_hand/reduce_10feature_datas.npy', allow_pickle=True)
datay = np.load(f"{BASE_URL}/feature_datas/feature_labels.npy")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG = CONFIG()
EPOCHS = CONFIG.EPOCHS
BATCH_SIZE = CONFIG.BATCH_SIZE

for i in range(3):
    
    trainx, valx, testx = myCV(datax, num_fold=3, val_fold=i)
    trainy, valy, testy = myCV(datay, num_fold=3, val_fold=i)

    
    #trainx , valx = train_test_split(datax, test_size=0.2, random_state=0)  #, test_size=0.2, train_size=0.8)
    #valx, testx = train_test_split(valx, test_size=0.5, train_size=0.5, random_state=0)
    #trainy, valy = train_test_split(datay, test_size=0.2, random_state=0)
    #valy, testy = train_test_split(valy, test_size=0.5, train_size=0.5, random_state=0)
    

    #train_index = list(range(len(trainx)))
    #np.random.shuffle(train_index)
    #trainx = trainx[train_index]
    #trainy = trainy[train_index]
    myshape(trainx, valx, testx)
    myshape(trainy, valy, testy)

    #print(trainx)

    train_data = ASLData(trainx, trainy)
    valid_data = ASLData(valx, valy)
    test_data = ASLData(testx, testy)
    #print(train_data)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)


    print("train_loader:", train_loader, len(train_loader))
    #model = ASLModel(p=CONFIG.DROPRATE, INPUT_LENGTH=trainx.shape[1]).to(device)
    #model = GetModel(flat_frame_len=trainx.shape[1]).to(device)

    #model = Conv3DModel().to(device)
    #model = RNN(input_size=246, hidden_size=512, num_layers=4, output_size = 250).to(device)  #(batch_size, sequence_length, input_size)
    #model = GRU().to(device)
    #model = My3DCNN().to(device)

    model1 = model2 = model0 = Conv1DGRUModel().to(device)

    #opt = torch.optim.Adam(model1.parameters(), lr=CONFIG.LR)
    checkpoint0 = torch.load(f"C:/Users/ryu91/kaggle/Google_ISLR_ASL/runs/cv/best_model0.pt")
    checkpoint1 = torch.load(f"C:/Users/ryu91/kaggle/Google_ISLR_ASL/runs/cv/best_model1.pt")
    checkpoint2 = torch.load(f"C:/Users/ryu91/kaggle/Google_ISLR_ASL/runs/cv/best_model2.pt")

    model0.load_state_dict(checkpoint0['model_state_dict'])
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model2.load_state_dict(checkpoint2['model_state_dict'])

    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']


    criterion = nn.CrossEntropyLoss()

    
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
    #Eval testdata
    #test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    test_loss_sum = 0.
    test_correct = 0
    test_total = 0
    ALLS = []

    model1.eval()
    model2.eval()
    model0.eval()
    for x,y in test_loader:
        x = torch.Tensor(x).float().to(device)
        y = torch.Tensor(y).long().to(device)

        with torch.no_grad():
            y_pred0 = model0(x)
            y_pred1 = model1(x)
            y_pred2 = model2(x)

            #print(y, y_pred0, y_pred1, y_pred2)
            #print(np.argmax(y_pred.cpu().numpy(), axis=1))
            #print(y.cpu().numpy())
            # 平均を計算する
            y_pred = torch.mean(torch.stack([y_pred0, y_pred1, y_pred2]), dim=0)

            loss = criterion(y_pred, y)
            test_loss_sum += loss.item()
            test_correct += np.sum((np.argmax(y_pred.cpu().numpy(), axis=1) == y.cpu().numpy()))
            test_total += 1

            ALLS.append([y, y_pred])


    #print(f"Epoch:{i} > Train Loss: {(train_loss_sum/train_total):.04f}, Train Acc: {train_correct/len(train_data):0.04f}")
    print(f"test Loss: {(test_loss_sum/test_total):.04f}, test Acc: {test_correct/len(test_data):0.04f}")


    with open('data.pickle', mode='wb') as f:
        pickle.dump(ALLS, f)

    i+=1
    print("@@@@@@@@"*5)