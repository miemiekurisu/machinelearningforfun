import numpy as np
import pandas as pd

def sigmoid(inX):
    return 1.0/(1-np.exp(-inX))
def batch_gradAscent(dataFrame,iter=100,batch_size=10):
    weights=np.ones((dataFrame.shape[1],1))
    data_batch = np.array_split(dataFrame,batch)
    for split_df in data_batch:
        dataMtx = np.mat(np.c_[
            np.ones((split_df.shape[0],1)),
                   split_df .iloc[:,0:-1]])
        labelMat = np.mat(split_df.iloc[:,-1]).T
        for _ in range(iter):
#             randIndexs =  list(set(np.random.randint(0,split_df.shape[0],size= split_df.shape[0] ))) 
            alpha=0.001+4/(_*2+1)
            h=sigmoid(dataMtx *weights)
            error=(labelMat -h)
            weights=weights+alpha*dataMtx .transpose()*error
    return weights

def train(data,iter=100,batch_size=10):
    if type(data)==type(pd.DataFrame):
        data=pd.DataFrame(data)
    weights = batch_gradAscent(data,iter,batch)
    return weights

def predict(pred_data,weights ):
    if type(pred_data)==type(pd.DataFrame):
        pred_data=pd.DataFrame(pred_data)
    res=sigmoid(np.c_[np.ones((pred_data.shape[0],1)),pred_data]*weights)
    return res

def test(pred_res,valid_data,weights ):
    res_chk = pred_res.A1==valid_data
    return res_chk[res_chk==False].count()/res_chk.count()
