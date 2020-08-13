"""
新規コロナウィルス感染者数を予測するプログラム

新規感染者数のデータを使用
都市部の人口密度を使用
"""
import os
import numpy as np
import pandas as pd

from pulp import *

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import datetime
import time

import warnings
warnings.filterwarnings("ignore")

# csvファイルの読み込み
DATA = pd.read_csv("Population.csv")

print("Population .shape ", DATA.shape)
DATA

# モデルの計算、計算結果の保存用のデータフレームにコピー
DF=DATA.copy()
DF['day']=DF['日付']
DF['last_year_density'] = DF['感染拡大以前との比較'].rolling(7).sum()/7 #感染拡大以前との比較
DF['last_year_density'] = DF['last_year_density']/-100 # %を変換
DF['yesterday_density'] = DF['前日との比較'].rolling(7).sum()/7 #前日との比較
DF['new_p'] = DF['新規感染者数'].rolling(7).sum()/7
DF

optim_num=20000    # 最適化計算をする回数 

X_list=np.array([[0,20, 0.6 , 5.910],[10,20 , 1.5 , 11],[8,30, 1.5, 13],[-1,10, 0.4 , 1.8705],[-1,10,0.4,0.84210]])
   # 最適化するパラメータ[a,N,M,r,s] の　範囲（最小、最大）と刻み、初期値
     
column_list=['new_p','last_year_density','inf']  # データフレーム中　使用する変数
#　最後のリストは予測値を入れる列

idx_last=73    # 最適化に使用するデータの最大値、最小値
idx_start=20   

class Optimizer:
    """
    # 新規感染者数を推測するモデルを学習させるクラス
    予測モデル、予測値を計算する。
    予測モデルを複数回十国することにより最適化を行う
        
    """
    def __init__(self,
                 DF = DF,                 
                 X_list=X_list,            
                 idx_last=idx_last,     
                 idx_start=idx_start,      
                 column_list=column_list,  
                 optim_num=optim_num       
                 ):
        """
        初期値の設定
        
        Argments:
            DF (object): 使用するデータフレーム
            X_list (list): 最適化するパラメータの範囲と刻み、初期値
            idx_start (int): 学習に使用するデータフレームの最初のidx
            idx_last (int): 学習に使用するデータフレームの最後のidx
            column_list (list): データフレームの中から使用するデータを決める列名リスト
            obtim_naum (list): 学習回数
        Returns:
            なし。
        """

        
        self.DF=DF
        self.X_list=X_list
        self.idx_start=idx_start
        self.idx_last=idx_last
        self.column_list=column_list
        self.optim_num=optim_num
        
        
    def Pred_y(self,X,idx):  
        """
        予測モデル
        
        Argments:
            inf (int): 感染者数項(3日分の平均)
            cont (int): 接触頻度項
            Pred_y (int): 予測値
        Returns:
            Pred_y (int): 予測値
        """
        inf = ((self.DF[self.column_list[0]].loc[idx-X[1]]+
                self.DF[self.column_list[0]].loc[idx-X[1]-1])+
               self.DF[self.column_list[0]].loc[idx-X[1]-2]/3)**X[4]  
        cont = self.DF[self.column_list[1]].loc[idx-X[2]]**X[3]      
        Pred_y = X[0] * inf * cont
    
        return Pred_y
    
    def calc_loss(self,X):    
    	 """
    	 指定範囲のデータを用いてモデルの予測誤差を計算
    	 
    	 Argments:
    	 	T (int): 実測値
    	 	P (int): 予測値
    	 	loss (int): 誤差
    	 Returns:
    	 	loss (int): 誤差
    	 """
    	 loss=0
    	 for idx in range(self.idx_start,self.idx_last):
        	T=self.DF[self.column_list[0]].loc[idx]  
        	P = self.Pred_y(X,idx)                   
        	loss += np.abs(T-P)
        	return loss/(self.idx_last-self.idx_start)
        	    
    def X_valid(self,X): 
        """
        乱数によりXの値を制約の中でシフトさせ、新たなXを発生させる
        
        Argments:
        	rand (int): -1~1の乱数を5個発生
        	dx (list): Xの変化幅
        	X (list):  X_listで定義された最大最小値内にクリップ
        Returns:
        	X (list): X_listで定義された最大最小値内にクリップ
        """
        rand=(np.random.rand(len(self.X_list)) -0.5)*2     
        dx=X_list[:,2]*rand                              
        dx[1:3]=np.round(dx[1:3])                        
        X=X+dx                                           
        X=np.clip(X,self.X_list[:,0],self.X_list[:,1])   
        return X
    
    def closing(self,X):    #
    	"""
    	終了処理、最終モデルの予測データの保存　グラフ化
    	
    	Argments:
    		DF (object): 学習済みモデルで予測した結果を保存
    		P (int): 予測値
    		DF_plot (object): プロット
    	Reutrns:
    		DF (object): 予測した結果を保存したデータフレーム
    	"""
    	self.DF[self.column_list[2]]='NaN'
    	for idx in range(self.idx_start,self.idx_last+int(min(X[1],X[2]))):# 学習済みモデルで予測・保存
            P = self.Pred_y(X,idx)
            self.DF[self.column_list[2]].loc[idx]=P
            
    	DF_plot=self.DF[self.column_list].loc[self.idx_start:self.idx_last]
    	DF_plot[self.column_list[1]]=DF_plot[self.column_list[1]]*500
    	DF_plot.plot()
    	return self.DF
        
        
    def optim(self):        
    	 """
    	 最適化処理
    	 
    	 Argments:
    	 	X (list): 初期値
    	 	X_min (list): 現在の誤差の最小値を保存
    	 	loss_min (list): 予測した結果の誤差
    	 	DF (object): 最適化処理されたデータをデータフレームに保存
    	 Returns:
    	 	X_min(list): 最適化されたときの最小の誤差
    	 	DF (object): 最適化されたデータを保存したデータフレーム
    	 """
    	 X=self.X_list[:,3]
    	 X_min=X
    	 loss_min=self.calc_loss(X)
    	 for i in range(self.optim_num):
    	 	   if i%5000==0:
    	 	   	    print('loss',loss_min,X_min,i)
    	 	   X=self.X_valid(X_min)
    	 	   loss = self.calc_loss(X)
    	 	   if loss<loss_min :
    	 	   	X_min=X
    	 	   	loss_min=loss
                #print('loss',loss,X,i)
    	 DF = self.closing(X_min)
    	 print('')
    	 print('loss',loss_min,X_min)
    	 return X_min , DF
        
# クラスを使い　モデルのパラメータを学習させる
Opti=Optimizer()
optim_X,DF00=Opti.optim()



# 学習したパラメータを入れたモデルを用い　感染者数の移動平均を予測する
column_list[1]='last_year_density'    # 使用する接触頻度特性

pred_last = 80 # 予測を行う　最終行数

Opti=Optimizer(DF=DF00)   # 新たなデータでクラスを立ち上げ直す
for idx in range(idx_last-1,pred_last):
    DF00['inf'].loc[idx]=Opti.Pred_y(optim_X,idx)    # 予測結果を保存
    DF00['new_p'].loc[idx]=Opti.Pred_y(optim_X,idx)  #　予測結果を予測に使う列に追加

#DF00[130:165]

#　新規感染者数の予測結果と　使用した接触頻度特性を表示
print('10 : ',DF00['day'].loc[10])
print('30 : ',DF00['day'].loc[30])
print('50: ',DF00['day'].loc[50])
print('70: ',DF00['day'].loc[70])

DF_plot=DF00[['inf',column_list[1]]].loc[idx_start:73]
DF_plot[column_list[1]]=DF_plot[column_list[1]]*500

DF_plot.plot()

#  感染者数を入れる列の追加
DF00['inf_2']='NaN'
DF00['T_inf']=DF00['新規感染者数']

# 移動平均から各曜日の値に戻す関数
monday_idx=74     # 月曜日のidx
coef_list=[0.593, 0.9237, 1.081, 1.112, 1.189 ,1.199, 0.851]   # 曜日ごとの係数　別途計算した
week=['月','火','水','木','金','土','日']

def calc_inf_num(idx):
    week_idx = (idx-monday_idx)% 7
    #print(week_idx)
    #print(week[week_idx])
    val=DF00['new_p'].loc[idx+3]*coef_list[week_idx]
    return val
    
#  元に戻す関数を使用し　感染者数の移動平均から 新規感染者数　累積感染者数を求める
lastID=78             # この処理を行う最後の行番号
last_T_PCR_p=idx_last+1 # 累計計算のスタート点  最後の累計計測値データ

for idx in range(idx_start,lastID):
    DF00['inf_2'].loc[idx]=int(calc_inf_num(idx))

#  感染者数の累積を計算する  
DF00['T_inf'].loc[last_T_PCR_p]=DF00['新規感染者数'].loc[last_T_PCR_p]   

for idx in range(idx_last+1,lastID):
    DF00['T_inf'].loc[idx]=DF00['T_inf'].loc[idx-1]+DF00['inf_2'].loc[idx]
    
#  感染者数の累積を表示する   
print('10 : ',DF00['day'].loc[10])
print('30 : ',DF00['day'].loc[30])
print('50 : ',DF00['day'].loc[50])
print('70 : ',DF00['day'].loc[70])


DF_plot=DF00['T_inf'].loc[66:73]
DF_plot.plot(label="新規感染者数")

DF_plot=DF00['inf_2'].loc[66:73]
DF_plot.plot(label="新規予測感染者数")

from sklearn.metrics import r2_score

# 予測したい日のデータをコピー
X = DF00[["T_inf"]]
X = pd.Series(X[1:]["T_inf"],dtype = "float64")
X = X[74:80]
Y = DF00[["inf_2"]]
Y = pd.Series(Y[1:]["inf_2"],dtype = "float64")
Y = Y.fillna(0)
Y = Y[74:80]
print(X)
print(Y)

print(r2_score(X,Y))
