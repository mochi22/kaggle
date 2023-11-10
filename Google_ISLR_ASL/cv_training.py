from main import *
import time

#def training(*args, **kwargs):
def cv_training(datax, datay, EPOCHS, BATCH_SIZE, do_wandb, device):
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

        model = Conv1DGRUModel().to(device)
        #model = model.to(device)

        print(next(model.parameters()).is_cuda)
        print(model)
        #print(next(iter(train_loader)))
        #Magic
        if do_wandb:
            wandb.watch(model) #log_freq=100)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
        criterion = nn.CrossEntropyLoss()
        #sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.95)
        #sched = OneCycleLR(opt, max_lr=0.1, total_steps=15000*EPOCHS , epochs=5, steps_per_epoch=len(train_loader))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min', factor=0.8, patience=2, verbose=True)

        earlystopping = cv_EarlyStopping(cv_fold=i, patience=CONFIG.PATIENCE)
        
        start = time.time()
        #training!
        cv_train(EPOCHS, BATCH_SIZE, model, train_loader, criterion, opt, sched, do_wandb, val_loader, train_data, valid_data, earlystopping, cv_fold=i)
        print("Learning Time:", time.time()-start)


        
        from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
        #Eval testdata
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
        test_loss_sum = 0.
        test_correct = 0
        test_total = 0
        ALLS = []
        model.eval()
        for x,y in test_loader:
            x = torch.Tensor(x).float().to(device)
            y = torch.Tensor(y).long().to(device)

            with torch.no_grad():
                y_pred = model(x)
                #print(np.argmax(y_pred.cpu().numpy(), axis=1))
                #print(y.cpu().numpy())
                loss = criterion(y_pred, y)
                test_loss_sum += loss.item()
                test_correct += np.sum((np.argmax(y_pred.cpu().numpy(), axis=1) == y.cpu().numpy()))
                test_total += 1

                ALLS.append([y, y_pred])


        #print(f"Epoch:{i} > Train Loss: {(train_loss_sum/train_total):.04f}, Train Acc: {train_correct/len(train_data):0.04f}")
        print(f"test Loss: {(test_loss_sum/test_total):.04f}, test Acc: {test_correct/len(test_data):0.04f}")

        import pickle
        f = open('test_list.txt', 'wb')
        pickle.dump(ALLS, f)
        f.close()

        i+=1
        print("@@@@@@@@"*5)


def training(datax, datay, EPOCHS, BATCH_SIZE, do_wandb, device):

    trainx, valx, testx = myCV(datax, num_fold=3, val_fold=2)
    trainy, valy, testy = myCV(datay, num_fold=3, val_fold=2)

    
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

    model = Conv1DGRUModel().to(device)
    #model = model.to(device)

    print(next(model.parameters()).is_cuda)
    print(model)
    #print(next(iter(train_loader)))
    #Magic
    if do_wandb:
        wandb.watch(model) #log_freq=100)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    criterion = nn.CrossEntropyLoss()
    #sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.95)
    #sched = OneCycleLR(opt, max_lr=0.1, total_steps=15000*EPOCHS , epochs=5, steps_per_epoch=len(train_loader))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min', factor=0.8, patience=2, verbose=True)

    earlystopping = EarlyStopping(patience=CONFIG.PATIENCE)
    
    start = time.time()
    #training!
    train(EPOCHS, BATCH_SIZE, model, train_loader, criterion, opt, sched, do_wandb, val_loader, train_data, valid_data, earlystopping)
    print("Learning Time:", time.time()-start)


    
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
    #Eval testdata
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    test_loss_sum = 0.
    test_correct = 0
    test_total = 0
    ALLS = []
    model.eval()
    for x,y in test_loader:
        x = torch.Tensor(x).float().to(device)
        y = torch.Tensor(y).long().to(device)

        with torch.no_grad():
            y_pred = model(x)
            #print(np.argmax(y_pred.cpu().numpy(), axis=1))
            #print(y.cpu().numpy())
            loss = criterion(y_pred, y)
            test_loss_sum += loss.item()
            test_correct += np.sum((np.argmax(y_pred.cpu().numpy(), axis=1) == y.cpu().numpy()))
            test_total += 1

            ALLS.append([y, y_pred])


    #print(f"Epoch:{i} > Train Loss: {(train_loss_sum/train_total):.04f}, Train Acc: {train_correct/len(train_data):0.04f}")
    print(f"test Loss: {(test_loss_sum/test_total):.04f}, test Acc: {test_correct/len(test_data):0.04f}")

    import pickle
    f = open('test_list.txt', 'wb')
    pickle.dump(ALLS, f)
    f.close()

    