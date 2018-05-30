# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
import operator as op



def C45_calcShannonEntGain(dataSet):
    if not type(dataSet)==type(pd.DataFrame):
        dataSet = pd.DataFrame(dataSet)
    fea_cols = dataSet.columns.tolist()[0:-1]
    tag_col = dataSet.columns.tolist()[-1]
    _ = dataSet[tag_col].groupby(dataSet[tag_col]).agg("count")/dataSet.shape[0]
    info_tag= (-_*np.log2(_)).sum()
    if info_tag==0:
        return info_tag,fea_cols[0]
    maxcol={}
    for _ in fea_cols:
        a=dataSet.groupby([_,tag_col]).size().reset_index(name='rcnt')
        b=dataSet.groupby([_]).size().reset_index(name='fcnt')
        h_=b['fcnt']/dataSet.shape[0]
        h = (- h_ * np.log2(h_)).sum()
        d1 = pd.merge(a, b, on=_,how='left')
        d1['ShannonStep']=(-d1['rcnt']/d1['fcnt']*np.log2(d1['rcnt']/d1['fcnt']))
        d1=d1.groupby([_,'fcnt'])['ShannonStep'].agg("sum").reset_index()
        d1['prob'] = d1["fcnt"]/dataSet.shape[0]
        gain = info_tag-(d1['prob']*d1['ShannonStep']).sum()
        maxcol[_] = gain/h
    return info_tag,sorted(maxcol.items(),key=op.itemgetter(1),reverse=True)[0][0]

def splitDF(dataSet,featurename):
    feature_value=dataSet[featurename].unique()
    f_col_idx = dataSet.columns.tolist().index(featurename)
    return dict(zip(feature_value,[dataSet[dataSet.iloc[:,f_col_idx]==_].drop([featurename],axis=1) for _ in feature_value]))

def buildTree(dataSet , tree={}):
    if not type(dataSet)==type(pd.DataFrame):
        dataSet = pd.DataFrame(dataSet)
    if dataSet.shape[1]<2:
        return dataSet.iloc[:,-1].unique()[0]
    else:
        pure,best_feature=C45_calcShannonEntGain(dataSet)
        t={best_feature:{}}
    if pure <= 0:
        return dataSet.iloc[:,-1].unique()[0]
    else:
        splited= splitDF(dataSet,best_feature)
        for k,v in splited.items():
            t[best_feature][k]=buildTree(v, t)
        return t