# -*- coding: utf-8 -*-
"""
新型コロナウイルスの新規感染者を予想するプログラム。
"""

import matplotlib.pyplot as plt
import datetime
import pandas as pd    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm  
from sklearn.metrics import r2_score
import warnings

def Dataset():
    """ 
    新規感染者数のデータセット。

    Args : 無し

    Returns : 
        kansensya:
            日本全体の１日ごとの新規感染者数のdataをPandasDataframe
            の型で返す.
            一列目に日付、二列目にその日の感染者数が入る。
    """

    #csvファイルの読み込み
    dataset = pd.read_csv("covid19.csv")
    #データ整形
    data_name = dataset.columns
    kansensya = pd.Series(dataset[1:]["日本国内新規罹患者数"],dtype = "float64")
    kansensya = kansensya.fillna(0)#NULL値の場合０を代入
    kansensya.index = pd.to_datetime(dataset[1:]["日付"], format='%Y/%m/%d')
    
    return kansensya


def predict_sarima(kansensya):
    """
    SARIMAmodelによる感染者予想をするプログラム。

    Args : 
        kansensya : 新規感染者数のデータセット（テストデータ)。

    Returns :
        sarimax_train_pred: SARIMAモデルによって予測された予測データ。
    """

    #総当たりによって最適なパラメーターを求めていく
    """
    warnings.filterwarnings('ignore') # 警告非表示（収束：ConvergenceWarning）
    # パラメータ範囲
    # order(p, d, q)
    min_p = 1; max_p = 3 # min_pは1以上を指定しないとエラー
    min_d = 0; max_d = 2
    min_q = 0; max_q = 3 

    # seasonal_order(sp, sd, sq)
    min_sp = 0; max_sp = 1
    min_sd = 0; max_sd = 1
    min_sq = 0; max_sq = 1

    test_pattern = (max_p - min_p +1)*(max_q - min_q + 1)*(max_d - min_d + 1)*(max_sp - min_sp + 1)*(max_sq - min_sq + 1)*(max_sd - min_sd + 1)
    print("pattern:", test_pattern)

    sfq = 7 # seasonal_order周期パラメータ
    ts = kansensya # 時系列データ

    test_results = pd.DataFrame(index=range(test_pattern), columns=["model_parameters", "aic"])
    num = 0
    for p in range(min_p, max_p + 1):
        for d in range(min_d, max_d + 1):
            for q in range(min_q, max_q + 1):
                for sp in range(min_sp, max_sp + 1):
                    for sd in range(min_sd, max_sd + 1):
                        for sq in range(min_sq, max_sq + 1):
                            sarima = sm.tsa.SARIMAX(
                                ts, order=(p, d, q), 
                                seasonal_order=(sp, sd, sq, sfq), 
                                enforce_stationarity = False, 
                                enforce_invertibility = False
                            ).fit()
                            test_results.iloc[num]["model_parameters"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), seasonal_order=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
                            test_results.iloc[num]["aic"] = sarima.aic
                            print(num,'/', test_pattern-1, test_results.iloc[num]["model_parameters"],  test_results.iloc[num]["aic"] )
                            num = num + 1
                            
                            

    # 結果（最小AiC）
    print("****************************")
    print("best[aic] parameter ********")
    print(test_results[test_results.aic == min(test_results.aic)])
    print("*******************************")
    """
    #テストデータと訓練データの作成
    kansensya_train = kansensya["2020-01-15":'2020-06-30'] 
    kansensya_test = kansensya["2020-07-01":"2020-07-07"]

    #求めた最適なパラメーターを選択して学習する
    sarimax_train = sm.tsa.SARIMAX(kansensya_train, 
                            order=(3, 2, 3),
                            seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity = False,
                            enforce_invertibility = False
                            ).fit()
    sarimax_train_pred = sarimax_train.predict("2020-07-01","2020-07-07")
    num = sarimax_train_pred._get_numeric_data()
    num[num<0] = 0.0#感染者数が負の数はあり得ないため負の数であった場合は０にする
    y_pred = sarimax_train_pred#予測値

    return [int(i)for i in sarimax_train_pred]


#都市での人口密度による　コロナ 予測
def Dataset2():
    """都市での人口密度によるコロナの感染者数の推移を行うためのデータセット。
    新宿駅のデータを読み込む。
    また今回のデータでは予測する日から７日前
    （５日前というのは、新型コロナウイルスの潜伏期間が７日だという過程のもとで行なっているため）
    のデータのさらに５日間の移動平均のデータを扱うので変形を行う。（平滑化目的)

    例）
        ６月３０日の学習データ（感染者数)　=>  6月１９日〜２５日の感染者数の平均をとる。

    Args :
        無し。

    Returns : 
        s_v :　データセット整形後のデータ。
    """
    #csvファイルの読み込み
    dataset = pd.read_csv("covid19.csv")    
    dataset2 = pd.read_csv("pop.csv")
    data_name2 = dataset.columns
    
    #新規感染者数の移動平均を取得
    move_mean = kansensya.copy()
    move_mean = move_mean.rolling(window = 7).mean()
    move_mean.name  = "mean"
    
    #日付データの整形
    pop = pd.Series(dataset2["感染拡大以前との比較"],dtype = "float64")
    pop2 = pd.Series(dataset2["前日との比較"],dtype = "float64").fillna(0)
    pop3 = pd.Series(dataset2["宣言前（７日)と比較"],dtype = "float64").fillna(0)

    pop.index  = pd.to_datetime("2020年"+dataset2["日付"], format='%Y年%m月%d日')
    pop2.index = pd.to_datetime("2020年"+dataset2["日付"], format='%Y年%m月%d日')
    pop3.index = pd.to_datetime("2020年"+dataset2["日付"], format='%Y年%m月%d日')

    #日付データとそれ以外のデータを５つずらす事で5日前のデータとして変形
    s_v = pd.concat([move_mean[90:],pop,pop2,pop3,kansensya[90:]],axis = 1)
    s_v["感染拡大以前との比較"] = s_v["感染拡大以前との比較"].shift(5)
    s_v["前日との比較"] = s_v["前日との比較"].shift(5)
    s_v["宣言前（７日)と比較"] = s_v["宣言前（７日)と比較"].shift(5)

    return s_v


def predict_Linear(dataset):
    """
    人口密度の増減に関するデータから新規感染者数を線形回帰を用いて感染者の予想を行う。
    線形回帰にはsklearn.linear_model の　LinearRegression　を用いているので、
    予測値　＝ w_0 + w_1 * x + w_2 * x_2 ...の式で定義されている。

    Args:
        dataset :
            datasetの中には,
                日本国内新規罹患者数（学習のテストデータ)
                mean : 7日前の感染者数の移動平均
                前日との比較　:　新宿駅のデータの人口密度の増減

    Returns :
        y_pred : 
            予測値。
    """
    
    s_v = dataset
    #訓練用データ   
    y = np.array(s_v["日本国内新規罹患者数"][8:-7])
    x = np.array(s_v[["mean","感染拡大以前との比較","前日との比較"]][8:-7])
    
    #説明変数の選択（どの説明変数を使ったらいい予測をするかを調べるときに使用したもの)
    #x_2 = np.array(s_v[["mean","感染拡大以前との比較"]][8:-7] )
    #x_3 = np.array(s_v[["mean","感染拡大以前との比較","前日との比較","宣言前（７日)と比較"]][8:-7] )
    #x_4 = np.array(s_v[["感染拡大以前との比較","前日との比較","宣言前（７日)と比較"]][8:-7] )

    #x_sample_test_2 = np.array(s_v[["mean","感染拡大以前との比較"]][-7:])
    #x_sample_test_3  = np.array(s_v[["mean","感染拡大以前との比較","前日との比較","宣言前（７日)と比較"]][-7:] )
    #x_sample_test_4 = np.array(s_v[["感染拡大以前との比較","前日との比較","宣言前（７日)と比較"]][-7:] )

    #テスト用データ（７月１日〜7日のデータ）
    x_sample_test = np.array(s_v[["mean","感染拡大以前との比較","前日との比較"]][-7:])
    y_sample = np.array(s_v["日本国内新規罹患者数"][-7:])

    #学習
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    model = LinearRegression()
    model.fit(x_train,y_train)

    #予測値
    y_pred = model.predict(x_sample_test)

    return [int(i) for i in y_pred] 


def getTestdata7_1_7():
    """
    ７月１日から７月７日までの新規感染者数のテストデータ
    予測値とのデータ評価はこの値を使って行う。

    Arg : 
        無し。
    Returns:
        nama_data :
            テスト用のデータ。
    """
    nama_data = [123,158,215,314,226,152,208]
    return nama_data 

#実行部分

kansensya = Dataset()
#print(kansensya)

#データ出力
print("SARIMAモデル予測値:",predict_sarima(kansensya))
print("新宿の人口密度による予測値:",predict_Linear(Dataset2()))
print("テストデータ:",getTestdata7_1_7())

#グラフ描画
plt.plot(getTestdata7_1_7(),label = "test_data(actual value)")
plt.plot(predict_sarima(kansensya),label = "SARIMA_model")
plt.plot(predict_Linear(Dataset2()),label = "Linear_model")
plt.legend()
plt.show()

#評価値の出力　決定係数
print("sarimaモデルの評価:",r2_score(getTestdata7_1_7(),predict_sarima(kansensya)))
print("新宿の人口密度による予想の評価:",r2_score(getTestdata7_1_7(),predict_Linear(Dataset2())))
print("決定係数による評価　(1が最高)")