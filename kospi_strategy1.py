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

raw_data = pd.read_csv('kospi_data_sample.csv',index_col='Date',parse_dates=True)
raw_data.sort_index(inplace=True)

window = 7   ## MA 계산할 주기 기본 7주
cost = 0.02  ## 수익률 계산할 때 보정치
holding_period = 3  ## 진입 이후 포지션 조정 없는 주간 수


##### 데이터 import 영역 끝 ########################################################################
#
# def _isBull(data, window):
#     ## 주가가 12주 이동평균선을 상향 교차하면 상승장이다
#     ## 주가가 12주 이동평균선을 하향 교차하면 하락장이다
#     ## 주가가 12주 이동평균선을 한 방향으로 교차한 지 10봉 이내에 반대 방향으로 교차하면 횡보장이다
#     ## 그러나 Trading용으로 여러 주기를 탐색해서 그동안 최적의 수익률을 보여왔던 주기를 탐색한다
#
#     ## set price variables
#
#     close_price = data.Close
#     high_price = data.High
#     open_price = data.Open
#     low_price = data.Low
#
#     ## 주간 이동평균선
#     wma = close_price.rolling(window).mean()
#     wma = wma.shift(1)
#     data_merged = pd.merge(close_price.shift(1),close_price,left_index=True,right_index=True)
#     data_merged = pd.merge(data_merged,wma,left_index=True, right_index=True)
#     data_merged.rename(columns={'Close_x':'Open',"Close_y":'Close',"Close":'MA'},inplace=True)
#
#     cross_date = []
#     # cross_date = data_merged[data_merged.MA < data_merged.Close]
#     # cross_date = cross_date[cross_date.MA > cross_date.Open]
#
#     for i in range(0,len(data_merged.index)):
#         cl_price = data_merged.iloc[i].Close
#         op_price = data_merged.iloc[i].Open
#         ma_price = data_merged.iloc[i].MA
#
#         if cl_price > ma_price and op_price < ma_price :
#             # print(data.iloc[0:i])
#             bear_list = _isBear(data.iloc[0:i],window)
#             if bear_list == [] :
#                cross_date = cross_date + [data_merged.index[i]]
#                # print(data_merged.index[i])
#             else :
#                 if (data_merged.index[i] - bear_list.index[-1]).days > 21 :
#                     cross_date = cross_date + [data_merged.index[i]]
#
#
#
#     return cross_date
#
# def _isBear(data, window):
#     ## 주가가 12주 이동평균선을 상향 교차하면 상승장이다
#     ## 주가가 12주 이동평균선을 하향 교차하면 하락장이다
#     ## 주가가 12주 이동평균선을 한 방향으로 교차한 지 10봉 이내에 반대 방향으로 교차하면 횡보장이다
#     ## 그러나 Trading용으로 여러 주기를 탐색해서 그동안 최적의 수익률을 보여왔던 주기를 탐색한다
#
#     ## set price variables
#
#     close_price = data.Close
#     high_price = data.High
#     open_price = data.Open
#     low_price = data.Low
#
#     ## 12주 이동평균선
#     wma = close_price.rolling(window).mean()
#     wma = wma.shift(1)
#     data_merged = pd.merge(close_price.shift(1),close_price,left_index=True,right_index=True)
#     data_merged = pd.merge(data_merged,wma,left_index=True, right_index=True)
#     data_merged.rename(columns={'Close_x':'Open',"Close_y":'Close',"Close":'MA'},inplace=True)
#
#
#     cross_date = []
#
#     for i in range(0,len(data_merged.index)):
#         cl_price = data_merged.iloc[i].Close
#         op_price = data_merged.iloc[i].Open
#         ma_price = data_merged.iloc[i].MA
#         if cl_price < ma_price and op_price > ma_price :
#             bull_list = _isBull(data.iloc[0:i],window)
#             print(data)
#             print("Bull list :",bull_list)
#             if bull_list == [] :
#                 print("bull list가 없습니다")
#                 cross_date = cross_date + [data_merged.index[i]]
#             else :
#                 print(data_merged[i])
#                 print(bull_list)
#                 if (data_merged.index[i] - bull_list.index[-1]).days > 21 :
#                     cross_date = cross_date + [data_merged.index[i]]
#
#
#
#     print(cross_date)
#     return cross_date
#
# ### Bull 신호가 연속으로 중복되거나 Bear 신호가 연속으로 중복되는 경우 제거
# def cleansing(bull_date,bear_date):
#     bear_date_list = []
#     bull_date_list = []
#
#     for i in range(0, len(bull_date.index) - 1):
#         axis_date = bull_date.index[i]
#         next_date = bull_date.index[i + 1]
#
#         end_date = bear_date[bear_date.index > axis_date].index[0]
#         try:
#             if bear_date_list[-1] != end_date:
#                 bear_date_list = bear_date_list+[end_date]
#         except:
#             if bear_date.index[0] < bull_date.index[0]:
#                 bear_date_list = [bear_date.index[0]] + [end_date]
#             else:
#                 bear_date_list = [end_date]
#
#     for i in range(0, len(bear_date.index) - 1):
#         axis_date = bear_date.index[i]
#         next_date = bear_date.index[i + 1]
#
#         end_date = bull_date[bull_date.index > axis_date].index[0]
#
#         try:
#             if bull_date_list[-1] != end_date:
#                 bull_date_list = bull_date_list+[end_date]
#         except:
#             if bull_date.index[0] < bear_date.index[0]:
#                 bull_date_list = [bull_date.index[0]] + [end_date]
#             else:
#                 bull_date_list = [end_date]
#
#     return bull_date_list, bear_date_list
#
# #### 포지션 조정 이후 일정 기간 동안 포지션 홀딩
# #### positon_holding
#
# def position_hold(bull_dates, bear_dates, holding_period):
#     print(len(bull_dates))
#     print(len(bear_dates))
#     i = 0
#     # while i < min(len(bull_dates),len(bear_dates)):
#     for i in range(0,min(len(bull_dates),len(bear_dates))):
#         try:
#             if (bear_dates[i] - bull_dates[i]).days < 7 * holding_period:
#                 del bear_dates[i]
#                 del bull_dates[i+1]
#                 i = i -1
#
#             if (bull_dates[i+1] - bear_dates[i]).days < 7 * holding_period:
#                 del bull_dates[i+1]
#                 del bear_dates[i+1]
#                 i = i -1
#         except:
#             pass
#
#     print(bull_dates)
#     print(bear_dates)
#     print(len(bull_dates))
#     print(len(bear_dates))
#

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

# def profit_cal(data,other_data,dates,other_dates,key = 1, cost=0.01):
#     profit_list = []
#     kospi_list = []
#
#     for i in range(0,min(len(dates),len(other_dates))):
#         start_price = float(data[data.index == dates[i]].MA)
#         end_price = float(other_data[other_data.index > dates[i]].Close[0])
#
#         if key == 1:
#             profit = (end_price / start_price) - 1 - cost #### 실질적으로 MA 돌파시에 바로 포지션을 바꿀 수 없기 때문에 2% 손해 가정
#         elif key == -1:
#             profit = -((end_price / start_price) -1)  - cost #### 실질적으로 MA 돌파시에 바로 포지션을 바꿀 수 없기 때문에 2% 손해 가정
#
#         profit_list = profit_list + [profit]
#         kospi_list = kospi_list + [data.Close[i]]
#
#     df = {'Kospi': kospi_list, 'profit': profit_list}
#     df = pd.DataFrame(df, index=dates[0:min(len(dates),len(other_dates))])
#
#     return df

def profit_cal(data,dates,other_dates,key=1):
    profit_list = []
    kospi_list =[]

    for i in range(0,min(len(dates),len(other_dates))):
        start_price = float(data[data.index == dates[i]].MA)
        end_price = float(data[data.index == other_dates[i]].MA)

        if key == 1:
            profit = (end_price/start_price) - 1 - 0.02

        profit_list = profit_list + [profit]
        kospi_list = kospi_list + [data.Close[i]]

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

print(bear_dates)
print(bull_dates)


print(profit_cal(data_merged,bull_dates,bear_dates,key=1))


# df1 = profit_cal(bull_date,bear_date,bull_dates,bear_dates,1, cost)
# df2 = profit_cal(bear_date,bull_date,bear_dates,bull_dates,-1, cost)
#
#
# df_mer = pd.merge(df1.profit,df2.profit,how='outer',left_index=True,right_index=True)
# df_mer.to_csv('kospi_strategy.csv')
# df_mer.fillna(0,inplace=True)
#
# df_mer = df_mer.profit_x+df_mer.profit_y
#
#
# profit_tr = 100.0
# profit_tr_list = []
# kospi_list =[]
#
# for i in range(0,len(df_mer.index)):
#     profit_tr = profit_tr*(1+df_mer[i])
#     profit_tr_list = profit_tr_list + [profit_tr]
#     kospi_list = kospi_list + [float(raw_data[raw_data.index==df_mer.index[i]].Close)]
#
#
#
# kospi_list = [x/kospi_list[0]*100 for x in kospi_list]
#
# df_profit = {'Profit':profit_tr_list,'KOSPI':kospi_list}
#
# df_profit = pd.DataFrame(df_profit)
# df_profit.index = df_mer.index


##### 데이터 조작 영역 끝 ###############################################################
#
# print("Total Return = ", profit_tr_list[-1])
# print("Benchmark Return = ",kospi_list[-1])
# print("MDD = ",round(df_mer.min()*100,2),"%")
# print("Mean = ",round(df_mer.mean()*100,2),"%")
# print("Hit Rataio = ",round(len(df_mer[df_mer>0])/len(df_mer),2))
#
# ax1 = plt.subplot(3,1,1)
# plt.plot(df_profit.index,df_profit.Profit,'y-')
# plt.title("Strategy Total Return")
# plt.ylabel("(2000Y = 100)")
# plt.yscale('log')
# print(ax1)
#
# ax2 = plt.subplot(3,1,2)
# plt.plot(df_profit.index,df_profit.KOSPI,'r--')
# plt.title("Benchmark Return")
# plt.xlabel('Dates')
# plt.ylabel("(2000Y = 100)")
# print(ax2)
#
# ax3 = plt.subplot(3,1,3)
# plt.hist(df_mer,bins=60)
# plt.title("Return histogram")
# plt.xlabel('Dates')
# plt.ylabel("unit")
# print(ax2)
#
# plt.tight_layout()
# plt.show()

##### 차트 및 시각화 영역 끝 ############################################################
