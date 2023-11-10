class CONFIG():
    EPOCHS = 50  #50
    LR = 0.000333  #0.005
    BATCH_SIZE = 32  #64
    DROPRATE = 0.20  #0.25
    RUN_NAME = 'feature_5dif_meanandstd_GRU_hiddendim128'
    SAVE_DIR = 'C:/Users/ryu91/kaggle/Google_ISLR_ASL/runs'
    do_wandb = False
    PATIENCE = 10