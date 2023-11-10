from main import *
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

def train(EPOCHS, BATCH_SIZE, model, train_loader, criterion, opt, sched, do_wandb, val_loader, train_data, valid_data, earlystopping):
    for i in range(EPOCHS):
        model.train()
        
        train_loss_sum = 0.
        train_correct = 0
        train_total = 0
        train_bar = train_loader
        for x,y in train_bar:
            x = torch.Tensor(x).float().cuda()
            y = torch.Tensor(y).long().cuda()

            y_pred = model(x)

            trainloss = criterion(y_pred, y)

            trainloss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss_sum += trainloss.item()
            train_correct += np.sum((np.argmax(y_pred.detach().cpu().numpy(), axis=1) == y.cpu().numpy()))
            train_total += 1
            #sched.step()
            if do_wandb:
                wandb.log({"train_loss": trainloss, "train_sum_loss": train_loss_sum, 'train_loss_mean': train_loss_sum / train_total})
            
        val_loss_sum = 0.
        val_correct = 0

        val_total = 0
        Min_Val_Loss = 999
        
        model.eval()
        for x,y in val_loader:
            x = torch.Tensor(x).float().cuda()
            y = torch.Tensor(y).long().cuda()
            
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if Min_Val_Loss > loss:
                    Min_Val_Loss = loss

                val_loss_sum += loss.item()
                val_correct += np.sum((np.argmax(y_pred.cpu().numpy(), axis=1) == y.cpu().numpy()))
                #val_mat = confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy())
                #recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None) 
                #precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None)
                #acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None)
                #f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None)
                val_total += 1
                if do_wandb:
                    wandb.log({"val_loss": loss, "val_sum_loss": val_loss_sum,"val_loss_mean:":val_loss_sum/val_total ,"LR": sched.optimizer.param_groups[0]['lr']})
        sched.step(val_loss_sum/val_total)
        print(f"Epoch:{i} > Train Loss: {(train_loss_sum/train_total):.04f}, Train Acc: {train_correct/len(train_data):0.04f}, LR: {sched.optimizer.param_groups[0]['lr']}")
        print(f"Epoch:{i} > Val Loss: {(val_loss_sum/val_total):.04f}, Val Acc: {val_correct/len(valid_data):0.04f}")
        #print(f"Epoch:{i} >>>>> 'ACC':{acc}, 'recall':{recall}, 'precision':{precision}, 'f1_score':{f1} ")
        #if do_wandb:
            #wandb.log({"val_Acc:":val_correct/len(valid_data), "ACC":acc, "recall":recall, "precision":precision, "f1_score":f1 })
        print("="*50)

        earlystopping((loss / BATCH_SIZE), model, i, opt) #callメソッド呼び出し
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break

    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': trainloss,
        }, './runs/last_model.pt')

    print("best_val_loss:", Min_Val_Loss)

    #VRAM解放
    gc.collect()



from main import *
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

def cv_train(EPOCHS, BATCH_SIZE, model, train_loader, criterion, opt, sched, do_wandb, val_loader, train_data, valid_data, earlystopping, cv_fold):
    for i in range(EPOCHS):
        model.train()
        
        train_loss_sum = 0.
        train_correct = 0
        train_total = 0
        train_bar = train_loader
        for x,y in train_bar:
            x = torch.Tensor(x).float().cuda()
            y = torch.Tensor(y).long().cuda()

            y_pred = model(x)

            trainloss = criterion(y_pred, y)

            trainloss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss_sum += trainloss.item()
            train_correct += np.sum((np.argmax(y_pred.detach().cpu().numpy(), axis=1) == y.cpu().numpy()))
            train_total += 1
            #sched.step()
            if do_wandb:
                wandb.log({"train_loss": trainloss, "train_sum_loss": train_loss_sum, 'train_loss_mean': train_loss_sum / train_total})
            
        val_loss_sum = 0.
        val_correct = 0

        val_total = 0
        Min_Val_Loss = 999
        
        model.eval()
        for x,y in val_loader:
            x = torch.Tensor(x).float().cuda()
            y = torch.Tensor(y).long().cuda()
            
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if Min_Val_Loss > loss:
                    Min_Val_Loss = loss

                val_loss_sum += loss.item()
                val_correct += np.sum((np.argmax(y_pred.cpu().numpy(), axis=1) == y.cpu().numpy()))
                #val_mat = confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy())
                #recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None) 
                #precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None)
                #acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None)
                #f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average=None)
                val_total += 1
                if do_wandb:
                    wandb.log({"val_loss": loss, "val_sum_loss": val_loss_sum,"val_loss_mean:":val_loss_sum/val_total ,"LR": sched.optimizer.param_groups[0]['lr']})
        sched.step(val_loss_sum/val_total)
        print(f"Epoch:{i} > Train Loss: {(train_loss_sum/train_total):.04f}, Train Acc: {train_correct/len(train_data):0.04f}, LR: {sched.optimizer.param_groups[0]['lr']}")
        print(f"Epoch:{i} > Val Loss: {(val_loss_sum/val_total):.04f}, Val Acc: {val_correct/len(valid_data):0.04f}")
        #print(f"Epoch:{i} >>>>> 'ACC':{acc}, 'recall':{recall}, 'precision':{precision}, 'f1_score':{f1} ")
        #if do_wandb:
            #wandb.log({"val_Acc:":val_correct/len(valid_data), "ACC":acc, "recall":recall, "precision":precision, "f1_score":f1 })
        print("="*50)

        earlystopping((loss / BATCH_SIZE), model, i, opt) #callメソッド呼び出し
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break

    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': trainloss,
        }, f'./runs/cv/last_model{cv_fold}.pt')

    print("best_val_loss:", Min_Val_Loss)

    #VRAM解放
    gc.collect()