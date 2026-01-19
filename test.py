
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 01:13:05 2023

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
import datetime
import xlwt
import warnings
from scipy.interpolate import interp1d
from dtaidistance import dtw

warnings.filterwarnings('ignore')

def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ln', type=bool, default=False,help='是否对油价取ln')
    parser.add_argument('--testLength', type=int, default=2,help='预测的个数')
    parser.add_argument('--test_index', type=int, default=1,help='预测第几个')
    args = parser.parse_args()
    return args

def getdata():
    #对数据进行预处理
    Brent_path =r"./data/Brent_multi_trend1.csv"
    data=pd.read_csv(Brent_path)
    return data

def splitdata(data,k,m,ln,test_index):
    """
    获取数据并得到历史、当前、预测数据集
    path:数据文件csv
    data_h,data_c是归一化后的
    data_f是原序列
    data_h_ori,data_c_ori是原序列
    """
    #获取历史数据，当前数据，预测数据
    test_start=data[data['date'] == "2024-12-24"].index[0]
    #进行第一个预测的时候，test_index=0，一直到row（0.2*len(data)）个,第一个是2019/4/2，test_index从1到1109
    now_index=test_start+test_index
    data_f=data.iloc[now_index:now_index+m,:]
    data_c_ori=data.iloc[now_index-k:now_index,:]
    data_h_ori=data.iloc[:now_index-k,:]
    return data_h_ori,data_c_ori,data_f

def dataTime(a,b):
    return a*b
dataTime = np.vectorize(dataTime)

def dataPlus(a,b):
    return a + b
dataPlus = np.vectorize(dataPlus)

def getX(ori_data_h, a):
    '''将历史模式使用a和b进行向量拉伸,输入dataframe返回拉伸后的dataframe'''
    b=1
    num_columns = ori_data_h.shape[1] - 1
    ori_data_h=ori_data_h.iloc[:,1:]
    M = int(len(ori_data_h) * a - a + 1)
    j = np.linspace(0, M-1 , M)
    t = (j + a) / a
    tmin = t.astype(int)
    tplu = tmin + 1
    AA = dataTime(np.repeat(np.array([t - tmin]),num_columns,axis=0).T,dataPlus(ori_data_h.iloc[np.where(tplu>len(ori_data_h),len(ori_data_h),tplu) - 1],- ori_data_h.iloc[tmin - 1]))
    xj = dataTime(b,dataPlus(ori_data_h.iloc[tmin - 1] , AA))
    xj = np.where(np.repeat(np.array([t - tmin]),num_columns,axis=0).T <= 0.000000000001,dataTime(b,ori_data_h.iloc[tmin - 1]),xj)
    return xj

def vector_bias(data_history,data_c):
    '''计算当前向量与历史向量匹配的偏差
        all_weight:获得每个变量及每个变量的权重
    '''
    #weight = np.array([0.6867,0.1900,0.1233])
    # weight = np.array([1,0,0])
    weight=[0.7933,0.1156,0.0911]
    data_c=data_c.astype(np.double)
    distance=[]
    for i in range(len(data_history)):
        data_h=data_history[i][0]
        data_h= data_h.astype(np.double)
        dtw_i = 0
        for dim in range(data_c.shape[1]):
            dtw_i += dtw.distance_fast(data_c[:,dim], data_h[:,dim],use_pruning=True)*weight[dim]
        distance.append(dtw_i)
    return distance

def gettopn(data_h_ori,data_c_ori,k):
    # 储存预测数据
    linerCount = 0
    all_bias=pd.DataFrame()
    train_ori=[]
    follow_ori=[]
    for aplus in list(range(5, 21)):
        a = aplus / 10
        if linerCount%5==0:
                print(linerCount,flush=True)
        linerCount+=1
        # 设置偏差列表
        offsets = []
        #trainpriceList变换历史价格
        trainPriceList = getX(data_h_ori, a)
        # 求训练列表中和测试列表中的差
        trainPrice = np.array([trainPriceList[i:i+k, :] for i in range(trainPriceList.shape[0] - k + 1 - m)])
        min_values = np.min(trainPrice, axis=1)
        min_values = np.repeat(min_values[:, np.newaxis, :], 10, axis=1)
        max_values = np.max(trainPrice, axis=1)
        max_values = np.repeat(max_values[:, np.newaxis, :], 10, axis=1)
        trainPrice = (trainPrice - min_values) / (max_values - min_values)
        data_c=np.array(data_c_ori)[:,1:]
        min_values = np.min(data_c, axis=0)
        max_values = np.max(data_c, axis=0)
        data_c = (data_c - min_values) / (max_values - min_values)
        trainPrice= np.split(trainPrice, trainPrice.shape[0])
        offsets=vector_bias(trainPrice, data_c)
        new_data= pd.DataFrame({'i': range(len(offsets)),'a':aplus,'bias': offsets})
        all_bias = all_bias.append(new_data, ignore_index=True)
        train_orilist=getX(data_h_ori, a)
        train_orilist = train_orilist[:, 0].tolist()
        train_ori+=[train_orilist[i:i+k] for i in range(len(train_orilist)-k+1-m)]
        follow_ori+=[train_orilist[i+k:i+k+m] for i in range(len(train_orilist)-k+1-m)]
    train_ori=np.array(train_ori)
    follow_ori=np.array(follow_ori)
    sorted_indices = all_bias['bias'].nsmallest(10).index
    top10_pattern_ori= train_ori[sorted_indices]
    top10_follow_ori= follow_ori[sorted_indices]
    min_values = np.min(top10_pattern_ori, axis=1)
    max_values = np.max(top10_pattern_ori, axis=1)
    follow_norm= (top10_follow_ori - min_values[:, np.newaxis]) / (max_values[:, np.newaxis] - min_values[:, np.newaxis])
    min_values_brent = np.min(data_c_ori['Brent'])
    max_values_brent = np.max(data_c_ori['Brent'])
    y_pred_norm= (follow_norm * (max_values_brent - min_values_brent)) + min_values_brent
    y_pred_norm_average= list(np.mean(np.array(y_pred_norm), axis=0))
    print('得到预测值',flush=True)
    return y_pred_norm_average

def run_sample(k,m,ln,test_index,testLength):
    '''
    只输出预测值
    '''
    currTime = str(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S"))
    print(f'当前时间为{currTime}',flush=True)
    data=getdata()
    preprices_norm_ave=[]
    test_start=data[data['date'] == "2019-4-2"].index[0]
    startPreDate=test_start+test_index
    ori_index=test_index
    ori_date=data.iloc[startPreDate,:]['date']
    ori_date='-'.join(ori_date.split('/'))
    outpath_norm_ave=os.path.join(r'./pre_norm_ave10_lightvpm/',f'{currTime}_predict{ori_date}_m{m}k_{k}_length{testLength}.xls')
    
    for i in range(testLength):
        #获取数据
        test_index=ori_index+i
        data_h_ori,data_c_ori,data_f=splitdata(data,k,m,ln,test_index)
        #寻找top10的i,a,b并给出预测值
        y_pred_norm_average=gettopn(data_h_ori,data_c_ori,k)
        preprices_norm_ave.append(y_pred_norm_average)
        print(f'已完成{i+1}次预测',flush=True)

    # 保存预测值
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('out')
    for i in range(len(preprices_norm_ave)):
        prePrice = preprices_norm_ave[i]
        worksheet.write(i, 0, label=data.iloc[startPreDate + i, :]['date'])
        for j in range(len(prePrice)):
            worksheet.write(i, j + 1, label=prePrice[j])
    workbook.save(outpath_norm_ave)
    print('预测值已保存',flush=True)

if __name__ == '__main__':
    m=10
    k=10
    ln=False
    testLength=6
    run_sample(k,m,ln,1439,2)
