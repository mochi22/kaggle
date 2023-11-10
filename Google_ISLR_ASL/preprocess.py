from main import *


def convert_row(row):
    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    #x = feature_converter(torch.tensor(x)).cpu().numpy()
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    #return x, row[1].label
    return x, row[1].label


def convert_and_save_data(Debug=False):
    df = pd.read_csv(TRAIN_FILE)
    if Debug:
        #df = df.iloc[:10]
        df = df.sample(10)
    df['label'] = df['sign'].map(label_map)
    npdata_x = np.zeros((df.shape[0], 10, 21, 3))
    nplabels = np.zeros(df.shape[0])
    #nplabels_face, nplabels_pose, nplabels_left, nplabels_right = np.zeros(2808), np.zeros(198), np.zeros(126), np.zeros(126)
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (x, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata_x[i,:] = x
            nplabels[i] = y
    np.save("reduce_feature_datas.npy", npdata_x)
    #np.save("feature_labels.npy", nplabels)

def convert_row_face(row):

    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    #x = feature_converter(torch.tensor(x)).cpu().numpy()
    xface = feature_converter(torch.tensor(x), face=True, pose=False, left = False, right = False).cpu().numpy()
    #return x, row[1].label
    return xface, row[1].label
def convert_row_pose(row):
    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    #x = feature_converter(torch.tensor(x)).cpu().numpy()
    xpose = feature_converter(torch.tensor(x), face=False, pose=True, left = False, right = False, lip = False).cpu().numpy()
    #return x, row[1].label
    return xpose, row[1].label
def convert_row_left(row):
    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    #x = feature_converter(torch.tensor(x)).cpu().numpy()
    xleft = feature_converter(torch.tensor(x), face=False, pose=False, left = True, right = False, lip = False).cpu().numpy()
    #return x, row[1].label
    return xleft, row[1].label
def convert_row_right(row):
    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    #x = feature_converter(torch.tensor(x)).cpu().numpy()
    xright = feature_converter(torch.tensor(x), face=False, pose=False, left = False, right = True, lip = False).cpu().numpy()
    #return x, row[1].label
    return xright, row[1].label
def convert_row_lip(row):
    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    #x = feature_converter(torch.tensor(x)).cpu().numpy()
    xlip = feature_converter(torch.tensor(x), face=False, pose=False, left = False, right = False, lip=True).cpu().numpy()
    #return x, row[1].label
    return xlip, row[1].label


"""
def myconvert_and_save_data(Debug=False):
    df = pd.read_csv(TRAIN_FILE)
    if Debug:
        df = df.iloc[:10]
    df['label'] = df['sign'].map(label_map)
    #npdata = np.zeros((df.shape[0], 3258))
    npdata_face = np.zeros((df.shape[0], 2808))
    npdata_pose = np.zeros((df.shape[0], 198))
    npdata_left = np.zeros((df.shape[0], 126))
    npdata_right = np.zeros((df.shape[0], 126))
    nplabels = np.zeros(df.shape[0])
    #nplabels_face, nplabels_pose, nplabels_left, nplabels_right = np.zeros(2808), np.zeros(198), np.zeros(126), np.zeros(126)
    with mp.Pool() as pool:
        results = pool.imap(convert_row_face, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (xface, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata_face[i,:] = xface
            nplabels[i] = y
    np.save("feature_data_face.npy", npdata_face)
    #np.save("feature_labels_face.npy", nplabels)
    with mp.Pool() as pool:
        results = pool.imap(convert_row_pose, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (xpose, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata_pose[i,:] = xpose
            nplabels[i] = y
    np.save("feature_data_pose.npy", npdata_pose)
    #np.save("feature_labels_pose.npy", nplabels)
    with mp.Pool() as pool:
        results = pool.imap(convert_row_left, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (xleft, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata_left[i,:] = xleft
            nplabels[i] = y
    np.save("feature_data_left.npy", npdata_left)
    #np.save("feature_labels_left.npy", nplabels)
    with mp.Pool() as pool:
        results = pool.imap(convert_row_right, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (xright, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata_right[i,:] = xright
            nplabels[i] = y
    np.save("feature_data_right.npy", npdata_right)
    #np.save("feature_labels_right.npy", nplabels)
"""


def myconvert_and_save_data(Debug=False):
    df = pd.read_csv(TRAIN_FILE)
    if Debug:
        df = df.sample(10)
    df['label'] = df['sign'].map(label_map)
    #npdata = np.zeros((df.shape[0], 3258))
    npdata_pose = np.zeros((df.shape[0], 198*4))
    #npdata_left = np.zeros((df.shape[0], 240)) #504))
    npdata_left = np.zeros((df.shape[0], 630))
    #npdata_right = np.zeros((df.shape[0], 504))
    npdata_right = np.zeros((df.shape[0], 630))
    #npdata_lip = np.zeros((df.shape[0], 240))#960))
    npdata_lip = np.zeros((df.shape[0], 1200))#960))
    nplabels = np.zeros(df.shape[0])
    #nplabels_face, nplabels_pose, nplabels_left, nplabels_right = np.zeros(2808), np.zeros(198), np.zeros(126), np.zeros(126)
    alls = []
    with mp.Pool() as pool:
        results = pool.imap(convert_row_left, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (x, y) in tqdm(enumerate(results), total=df.shape[0]):
            
            npdata_left[i,:] = x
            nplabels[i] = y

    np.save("dif_feature_data_left.npy", npdata_left)

    with mp.Pool() as pool:
        results = pool.imap(convert_row_right, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (x, y) in tqdm(enumerate(results), total=df.shape[0]):
            
            npdata_right[i,:] = x
            nplabels[i] = y

    np.save("dif_feature_data_right.npy", npdata_right)
    #np.save("feature_labels_left.npy", nplabels)

    with mp.Pool() as pool:
        results = pool.imap(convert_row_lip, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (x, y) in tqdm(enumerate(results), total=df.shape[0]):
            
            npdata_lip[i,:] = x
            nplabels[i] = y

    np.save("dif_feature_data_lip.npy", npdata_lip)
    #np.save("feature_labels_left.npy", nplabels)


def time_convert_row(row):

    x = load_relevant_data_subset(os.path.join(f"{BASE_URL}/asl-signs", row[1].path))
    x = time_feature_converter(torch.tensor(x)).cpu().numpy()
    #return x, row[1].label
    return x, row[1].label


def time_convert_and_save_data(Debug=False):
    df = pd.read_csv(TRAIN_FILE)
    if Debug:
        df = df.iloc[:10]
    df['label'] = df['sign'].map(label_map)
    npdata_x = np.zeros((df.shape[0], 4887))
    nplabels = np.zeros(df.shape[0])
    #nplabels_face, nplabels_pose, nplabels_left, nplabels_right = np.zeros(2808), np.zeros(198), np.zeros(126), np.zeros(126)
    with mp.Pool() as pool:
        results = pool.imap(time_convert_row, df.iterrows(), chunksize=250)
        #for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
        for i, (x, y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata_x[i,:] = x
            nplabels[i] = y
    np.save("time_feature_datas.npy", npdata_x)
    #np.save("feature_labels.npy", nplabels)




right_handed_signer = [26734, 28656, 25571, 62590, 29302, 
                       49445, 53618, 18796,  4718,  2044, 
                       37779, 30680]
left_handed_signer  = [16069, 32319, 36257, 22343, 27610, 
                       61333, 34503, 55372, ]
both_hands_signer   = [37055, ]

messy = [29302, ]

QUICK_TEST = False
QUICK_LIMIT = 200

def shoulder_convert_and_save_data(INPUT_SHAPE, SEGMENTS):
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    total = df.shape[0]
    if QUICK_TEST:
        total = QUICK_LIMIT
    npdata = np.zeros((total, INPUT_SHAPE[0]*INPUT_SHAPE[1] + (SEGMENTS+1)*INPUT_SHAPE[1]*2))
    nplabels = np.zeros(total)
    for i, row in tqdm(enumerate(df.iterrows()), total=total):
        (x,y) = convert_row(row)
        npdata[i,:] = x
        nplabels[i] = y
        if QUICK_TEST and i == QUICK_LIMIT - 1:
            break
    
    np.save("shoulder_data.npy", npdata)
    #np.save("feature_labels.npy", nplabels)
        