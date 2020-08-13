'''
Prediction of the number of new infections COVID-19 using SEIR model
Optimize parameters using minimize@scipy

Used datasets
https://github.com/kaz-ogiwara/covid19/blob/master/data/summary.csv
Copyright (c) 2020 TOYO KEIZAI ONLINE
'''

#include package
import numpy as np
import pandas as pd
import datetime
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#data input from csv
EST_PERIOD = 14
TEST_PERIOD = 7
csv_input = pd.read_csv(filepath_or_buffer="summary.csv", encoding="ms932", sep=",")
now = len(csv_input)
latest = str(csv_input.values[now-1,0])+"/"+str(csv_input.values[now-1,1])+"/"+str(csv_input.values[now-1,2])
latest_date = datetime.datetime.strptime(latest,'%Y/%m/%d').date()
est_start_date = latest_date - datetime.timedelta(days=EST_PERIOD+TEST_PERIOD)
est_end_date = est_start_date + datetime.timedelta(days=EST_PERIOD)
test_start_date = est_end_date + datetime.timedelta(days=1)
test_end_date = latest_date

def seir_eq(v,t,beta,lp,ip):
    """
    Define differencial equation of seir model

    Parameters:
    ----------
    v : numpy.ndarray
        [v[0], v[1], v[2]] = [S, E, I]
    t : numpy.ndarray 
        List of small changes in the estimation period
    beta : int
        Infectious rate
    lp : int
        Latency period
    ip : int
        Infectious period

    Returns:
    ----------
    [s, e, i, r] : list of int
        s -> Differencial equation of susceptible
        e -> Differencial equation of incubation
        i -> Differencial equation of infectious
        r -> Differencial equation of removed
    """
    N=126100000-Discharged-Death
    s = -beta*v[0]*v[2]/N
    e = beta*v[0]*v[2]/N-(1/lp)*v[1]
    i = (1/lp)*v[1]-(1/ip)*v[2]
    r = (1/ip)*v[2]
    return [s,e,i,r]

def input_data(csv_input, reg_period, test_period):
    """
    Append number of new infections to list

    Parameters:
    ----------
    csv_input : pandas.core.frame.DataFrame
        Use pd.read_csv to read CSV file
    reg_period : int 
        Estimation period
    test_period : int
        Test period

    Returns:
    ----------
    data_covid : list of int
        Number of new infections per day
    """
    data_covid = []
    for k in range(now-(reg_period+test_period),now-test_period,1):
        data_covid.append(int(csv_input.values[k,3])-int(csv_input.values[(k-1),3]))
    print(data_covid)
    return (data_covid)

data_covid_reg = input_data(csv_input, EST_PERIOD, TEST_PERIOD)
data_covid_test = input_data(csv_input, TEST_PERIOD, 0)

#Assignment initial value for estimation
Discharged = int(csv_input.values[(now-EST_PERIOD-1),7])
Death = int(csv_input.values[(now-EST_PERIOD-1),8])
Positive = data_covid_reg[0]
N,S0,E0,I0=126100000-(Positive+Discharged+Death),0,Positive,(Discharged+Death)
ini_state=[N,S0,E0,I0]
#Assignment initial value for test
Discharged_T = int(csv_input.values[now-1,7])
Death_T = int(csv_input.values[now-1,8])
Positive_T = data_covid_test[0]
N_T,S0_T,E0_T,I0_T=126100000-(Positive_T+Discharged_T+Death_T),0,Positive_T,(Discharged_T+Death_T)
ini_state_t=[N_T,S0_T,E0_T,I0_T]

beta,lp,ip=1,1,1
dt = 0.01
t = np.arange(0,EST_PERIOD,dt)
t_t = np.arange(0,TEST_PERIOD,dt)

def estimate_i(ini_state,beta,lp,ip):
    """
    Function which estimate i from seir model func

    Parameters:
    ----------
    ini_state : numpy.ndarray
        [ini_state[0], ini_state[1], ini_state[2]] = [S, E, I]
    beta : int
        Infectious rate
    lp : int
        Latency period
    ip : int
        Infectious period

    Returns:
    ----------
    est[:,2] : numpy.ndarray
        Evaluation result of calculating differential equation
    """
    v=odeint(seir_eq,ini_state,t,args=(beta,lp,ip))
    est=v[0:int(EST_PERIOD/dt):int(1/dt)]
    return est[:,2]

def y(params):
    """
    Define logscale likelihood function

    Parameters:
    ----------
    params : list of int
        Parameters during optimization
        [params[0], params[1], params[2]] = [beta, lp, ip]        

    Returns:
    ----------
    np.sum(est_i-data_covid_reg*np.log(np.abs(est_i))) : numpy.ndarray
        Logscale likelihood function
    """
    est_i=estimate_i(ini_state,params[0],params[1],params[2])
    return np.sum(est_i-data_covid_reg*np.log(np.abs(est_i)))

#optimize logscale likelihood function
mnmz=minimize(y,[beta,lp,ip],method="nelder-mead")
print(mnmz)

#def R0(mnmz):
beta_const,lp,gamma_const = mnmz.x[0],mnmz.x[1],mnmz.x[2] #Infectious rate, Latency period, Infectious period
print(beta_const,lp,gamma_const)
R0 = beta_const*(1/gamma_const)
print(R0)

t_n=np.arange(0,EST_PERIOD+TEST_PERIOD,dt)
#plot reult with observed data
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.6180 * 4, 4*2))
lns1=ax1.plot(data_covid_reg,"o", color="red",label = "data")
lns2=ax1.plot(estimate_i(ini_state,mnmz.x[0],mnmz.x[1],mnmz.x[2]), label = "estimation")
lns_ax1 = lns1+lns2
labs_ax1 = [l.get_label() for l in lns_ax1]
ax1.legend(lns_ax1, labs_ax1, loc=0)
ax1.set_title('Reg:'+est_start_date.strftime("%Y/%m/%d")+'-'+est_end_date.strftime("%Y/%m/%d")+', Test:'+test_start_date.strftime("%Y/%m/%d")+'-'+test_end_date.strftime("%Y/%m/%d"))

data_covid = data_covid_reg + data_covid_test
reg_fin = odeint(seir_eq,ini_state_t,t_t,args=(mnmz.x[0],mnmz.x[1],mnmz.x[2]))[::100,2]
print(np.round(reg_fin))
print("平均絶対誤差",mean_absolute_error(data_covid_test, reg_fin))
print("決定係数",r2_score(data_covid_test, reg_fin))

lns3=ax2.plot(data_covid_test,"o", color="red",label = "data")
lns4=ax2.plot(t_t,odeint(seir_eq,ini_state_t,t_t,args=(mnmz.x[0],mnmz.x[1],mnmz.x[2])))
ax2.legend(['data','Susceptible','Exposed','Infected','Recovered'], loc=0)
ax2.set_title('SEIR_b{:.2f}_ip{:.2f}_gamma{:.2f}_N{:d}_E0{:d}_I0{:d}_R0{:.2f}'.format(beta_const,lp,gamma_const,N,E0,I0,R0))
plt.ylim(0, 1000)
#plt.savefig('./fig/SEIR_b{:.2f}_ip{:.2f}_gamma{:.2f}_N{:d}_E0{:d}_I0{:d}_R0{:.2f}_.png'.format(beta_const,lp,gamma_const,N,E0,I0,R0)) 
plt.show()
plt.close()