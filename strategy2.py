## KOSPI Backtesting code
## This is to backtest for following strategy :

## KOSPI 진입
## 타임프레임 : 6개월~9개월
## 거래 종목 : KOSPI 2X 레버리지
## 거래차트 : 일봉, 주봉
## 진입 이유 : 1. 원화 가치 하락에 따라 수출 증가 가능, 2. 국내 경기 둔화에 따른 정부의 재정 지출 확대. 3. 최근 하락세에 따른 밸류에이션 매력 증가
## 진입 전략 : 7주 이평선을 장대 양봉으로 돌파한 순간 진입
## 청산 전략 : 시장이 장대 양봉 돌파 순간의 최저점을 '최저 저항선'으로 지정하고 가격이 최저 저항선 하래로 내려가지 않는 상황에서는 3개월간 보유


#####################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_data = pd.read_csv('kospi_data.csv',index_col='Date',parse_dates=True)
# raw_data = raw_data.iloc[:1000]
raw_data.sort_index(inplace=True)

window = 7   ## MA 계산할 주기 기본 7주
holding_period = 1  ## 진입 이후 포지션 조정 없는 개월 수
cost= 0.005



##### 데이터 import 영역 끝 ########################################################################
def data_merge(data,window):
    ## set price variables

    close_price = data.Close
    high_price = data.High
    open_price = data.Open
    low_price = data.Low

    ## 12주 이동평균선 (1주 선행)
    wma = close_price.rolling(window).mean()
    wma = wma.shift(1)
    data_merged = pd.merge(open_price, close_price, left_index=True, right_index=True)
    data_merged = pd.merge(data_merged, high_price, left_index=True, right_index=True)
    data_merged = pd.merge(data_merged, low_price, left_index=True, right_index=True)
    data_merged = pd.merge(data_merged, wma, left_index=True, right_index=True)
    data_merged.rename(columns={'Close_x': 'Close', "Close_y": 'MA'}, inplace=True)

    return data_merged

def pilar(data_merged):
    ### 장대양봉이고 7주 이평선이 장대 양봉 사이를 지나는지 확인하는 함수 ####

    # print(data_merged)
    data_pilar = []

    for i in range(0,len(data_merged.index)):
        if data_merged.Close[i]-data_merged.Open[i] > (data_merged.High[i]-data_merged.Low[i])*0.5:
           if  data_merged.MA[i] > data_merged.Open[i] and data_merged.MA[i] < data_merged.Close[i]:
                data_pilar = data_pilar + [data_merged.index[i]]

    return data_pilar

def least_resistance(data, date):
    ### 최소 저항선 설정 #####
    ### 일반적으로 장대 양봉이 7주 이평선에 도달한 이후 양봉의 최저점을 최저 저항선으로 설정
    lr_list = []

    for i in range(0,len(date)):
        lr = data[data.index==date[i]].Low
        lr_list = lr_list + [float(lr)]

    return lr_list

def searchFalseSignal(data):
    ### Moving Average 선을 돌파한 이후 1% 이상 상승하였지만 Close 가격이 다시 Moving Average 선 아래로 내려가는 경우를 False Signal로 정의
    ### 1% 선까지 올라갔다가 다시 MA 선을 하향 돌파하는 순간 포지션 정리. 이 경우 Profit에서 -1% 로 기록
    falsesignal =[]
    for i in range(0,len(data)):
        if data.Close[i] < data.MA[i] and data.Open[i] < data.MA[i] and data.High[i] > data.MA[i] :
            if data.High[i]/data.MA[i] > 1.01:
                falsesignal = falsesignal+[data.index[i]]

    return falsesignal

##### 함수 영역 끝 ###################################################################################

data_mer = data_merge(raw_data,window)
pilar_date = pilar(data_mer)
lr_data = least_resistance(raw_data,pilar_date)
false_date = searchFalseSignal(data_mer)

tt = 0
for i in pilar_date:
    if (data_mer.index[-1] - i).days < 90:
      tt += 1

le = len(pilar_date) - tt
print(pilar_date)

profit_list = np.repeat(0.0,le)
location = np.repeat(0,le)
time_frame = 4*holding_period

for i in range(0,le):
    key = False
    for j in range(0,time_frame):
        location[i] = len(data_mer[data_mer.index<=pilar_date[i]].index)
        if data_mer.iloc[location[i]+j].Low < lr_data[i]:
            profit = (lr_data[i]*(1-cost))/(data_mer[data_mer.index==pilar_date[i]].MA*(1+cost+0.005)) -1
            profit_list[i] = float(profit)
            key = True

    if key is False:
        profit = (data_mer.iloc[location[i]+time_frame-1].Close*(1-cost))/(data_mer.iloc[location[i]-1].MA*(1+cost))-1
        profit_list[i] = float(profit)

tr_profit = 100.0
tr_profit_list = []

for i in profit_list:
    tr_profit = tr_profit * (1+i)
    tr_profit_list = tr_profit_list + [tr_profit]

exit_date_list = []
for i in location:
    exit_date_list = exit_date_list +  [data_mer.iloc[i+time_frame-1].name]

benchmark_price_list =[]
for i in exit_date_list:
    benchmark_price = float(data_mer[data_mer.index == i].Close)
    benchmark_price_list = benchmark_price_list + [benchmark_price]

benchmark_price_list = [x/benchmark_price_list[0]*100 for x in benchmark_price_list]

df = {"Kospi":benchmark_price_list,"profit":tr_profit_list}
df = pd.DataFrame(df,index=exit_date_list)

##### 데이터 조작 영역 끝 ###############################################################

print("MA weeks = ", window)
print("hodling period =", holding_period,"months")
print("Total Return = ", tr_profit_list[-1])
print("Benchmark Return = ",benchmark_price_list[-1])
print("MDD = ",round(profit_list.min()*100,2),"%")
print("Hit Rataio = ",round(len(profit_list[profit_list>0])/len(profit_list),2))
print("Expected return = ",profit_list.mean())

# plt.hist(profit_list,bins=30)
# plt.show()

ax1 = plt.subplot(3,1,1)
plt.plot(df.index,df.profit,'y-')
plt.title("Strategy Total Return")
plt.ylabel("(2000Y = 100)")
# plt.yscale('log')
print(ax1)

ax2 = plt.subplot(3,1,2)
plt.plot(data_mer.index,data_mer.Close,'r-')
plt.title("Benchmark Return")
plt.xlabel('Dates')
plt.ylabel("(2000Y = 100)")
print(ax2)

ax3 = plt.subplot(3,1,3)
plt.hist(profit_list,bins=60)
plt.title("Return histogram")
plt.xlabel('Dates')
plt.ylabel("unit")
print(ax2)

plt.tight_layout()
plt.show()


##### 차트 및 시각화 영역 끝 ############################################################
