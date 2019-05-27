## KOSPI Backtesting code
## This is to backtest for following strategy :

## KOSPI 진입
## 타임프레임 : 6개월~9개월
## 거래 종목 : KOSPI 2X 레버리지
## 거래차트 : 일봉, 주봉
## 진입 이유 : 1. 원화 가치 하락에 따라 수출 증가 가능, 2. 국내 경기 둔화에 따른 정부의 재정 지출 확대. 3. 최근 하락세에 따른 밸류에이션 매력 증가
## 진입 전략 : 주봉 기준 하락세 종료 후 익일 시초가 매수
## 청산 전략 : 주봉 기준 상승세 종료 후 익일 시초가 매도

#####################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_data = pd.read_csv('kospi_data.csv',index_col='Date',parse_dates=True)
raw_data.sort_index(inplace=True)

window = 7   ## MA 계산할 주기 기본 7주
cost = 0.02  ## 수익률 계산할 때 보정치
holding_period = 3  ## 진입 이후 포지션 조정 없는 주간 수


##### 데이터 import 영역 끝 ########################################################################

def _dataMerged(data,window):
    ## set price variables

    close_price = data.Close
    high_price = data.High
    open_price = data.Open
    low_price = data.Low

    ## 12주 이동평균선
    wma = close_price.rolling(window).mean()
    wma = wma.shift(1)
    data_merged = pd.merge(close_price.shift(1), close_price, left_index=True, right_index=True)
    data_merged = pd.merge(data_merged, wma, left_index=True, right_index=True)
    data_merged.rename(columns={'Close_x': 'Open', "Close_y": 'Close', "Close": 'MA'}, inplace=True)

    return data_merged

def _checkCross(data_merged):
    ## 주가가 12주 이동평균선을 상향 교차하면 상승장이다
    ## 주가가 12주 이동평균선을 하향 교차하면 하락장이다
    ## 주가가 12주 이동평균선을 한 방향으로 교차한 지 10봉 이내에 반대 방향으로 교차하면 횡보장이다
    ## 그러나 Trading용으로 여러 주기를 탐색해서 그동안 최적의 수익률을 보여왔던 주기를 탐색한다

    cross_date = np.repeat(0,len(data_merged.index))

    for i in range(0,len(data_merged.index)):
        # print(data_merged.iloc[i])
        if data_merged.iloc[i].Open < data_merged.iloc[i].MA and data_merged.iloc[i].Close > data_merged.iloc[i].MA:
            cross_date[i] = 1
        if data_merged.iloc[i].Open > data_merged.iloc[i].MA and data_merged.iloc[i].Close < data_merged.iloc[i].MA:
            cross_date[i] = -1

    return cross_date

def _checkDateSpace(dateData):
    for i in range(0,len(dateData)-1):
        if dateData[i] != 0 :
            if dateData[i-1] !=0 or dateData[i-2] !=0 or dateData[i-3] != 0 :
                if dateData[i+1] == 0:
                    dateData[i+1] = dateData[i]
                    dateData[i] = 0
                else:
                    dateData[i] = 0

    return dateData

def _reduceDouble(dateData):
    k = 0
    for i in range(0,len(dateData)):
        if dateData[i] != 0 and k == 0:
            k = dateData[i]
        elif dateData[i] != 0 and k != 0:
            if dateData[i] == k :
                dateData[i] = 0
            else:
                k = dateData[i]

    return dateData

def profit_cal(data,dates,other_dates,cost,key=1):
    profit_list = []
    kospi_list =[]
    profit = 0.00

    for i in range(0,min(len(dates),len(other_dates))):
        start_price = float(data[data.index == dates[i]].Close)
        if dates[i] < other_dates[i]:
            end_price = float(data[data.index == other_dates[i]].Close)
        elif dates[i] > other_dates[i]:
            if i == min(len(dates),len(other_dates))-1:
                end_price = float(data.MA[len(data.index)-1])
            else:
                end_price = float(data[data.index == other_dates[i+1]].Close)

        if key == 1:
            profit = (end_price/start_price) - 1 - 0.005
        elif key == -1:
            profit = -((end_price/start_price) - 1) - 0.005

        profit_list = profit_list + [profit]
        kospi_list = kospi_list + [float(data[data.index==dates[i]].Close)]

    df = {'Kospi':kospi_list,'profit':profit_list}
    df = pd.DataFrame(df, index=dates[0:min(len(dates),len(other_dates))])

    return df

##### 함수영역 끝 ########################################################

data_merged = _dataMerged(raw_data,window)
cross_date_afterCheck = _checkCross(data_merged)
cross_date_with_Space = _checkDateSpace(cross_date_afterCheck)
final_dateData = _reduceDouble(cross_date_with_Space)


bull_dates = raw_data.iloc[final_dateData==1].index
bear_dates = raw_data.iloc[final_dateData==-1].index


print(bull_dates)
print(bear_dates)

df1 = profit_cal(data_merged,bull_dates,bear_dates,cost,key=1)
df2 = profit_cal(data_merged,bear_dates,bull_dates,cost,key=-1)


df_mer = pd.merge(df1.profit,df2.profit,how='outer',left_index=True,right_index=True)
df_mer.to_csv('kospi_strategy.csv')
df_mer.fillna(0,inplace=True)

df_mer = df_mer.profit_x+df_mer.profit_y


profit_tr = 100.0
profit_tr_list = []
kospi_list =[]

for i in range(0,len(df_mer.index)):
    profit_tr = profit_tr*(1+df_mer[i])
    profit_tr_list = profit_tr_list + [profit_tr]
    kospi_list = kospi_list + [float(raw_data[raw_data.index==df_mer.index[i]].Close)]



kospi_list = [x/kospi_list[0]*100 for x in kospi_list]

df_profit = {'Profit':profit_tr_list,'KOSPI':kospi_list}

df_profit = pd.DataFrame(df_profit)
df_profit.index = df_mer.index


##### 데이터 조작 영역 끝 ###############################################################

print("Total Return = ", profit_tr_list[-1])
print("Benchmark Return = ",kospi_list[-1])
print("MDD = ",round(df_mer.min()*100,2),"%")
print("Mean = ",round(df_mer.mean()*100,2),"%")
print("Hit Rataio = ",round(len(df_mer[df_mer>0])/len(df_mer),2))

ax1 = plt.subplot(3,1,1)
plt.plot(df_profit.index,df_profit.Profit,'y-')
plt.title("Strategy Total Return")
plt.ylabel("(2000Y = 100)")
# plt.yscale('log')
print(ax1)

ax2 = plt.subplot(3,1,2)
plt.plot(df_profit.index,df_profit.KOSPI,'r--')
plt.title("Benchmark Return")
plt.xlabel('Dates')
plt.ylabel("(2000Y = 100)")
print(ax2)

ax3 = plt.subplot(3,1,3)
plt.hist(df_mer,bins=60)
plt.title("Return histogram")
plt.xlabel('Dates')
plt.ylabel("unit")
print(ax2)

plt.tight_layout()
plt.show()

##### 차트 및 시각화 영역 끝 ############################################################
