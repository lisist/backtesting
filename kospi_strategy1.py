## KOSPI Backtesting code
## This is to backtest for following strategy :

## KOSPI 진입
## 타임프레임 : 6개월~9개월
## 거래 종목 : KOSPI 2X 레버리지
## 거래차트 : 일봉, 주봉
## 진입 이유 : 1. 원화 가치 하락에 따라 수출 증가 가능, 2. 국내 경기 둔화에 따른 정부의 재정 지출 확대. 3. 최근 하락세에 따른 밸류에이션 매력 증가
## 진입 전략 : 주봉 기준 하락세 종료 후 익일 시초가 매수
## 청산 전략 : 주봉 기준 상승세 종료 후 익일 시초가 매도

##

import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('kospi_data.csv',index_col='Date',parse_dates=True)
raw_data.sort_index(inplace=True)

def _isBull(data):
    ## 주가가 7주 이동평균선을 상향 교차하면 상승장이다
    ## 주가가 7주 이동평균선을 하향 교차하면 하락장이다
    ## 주가가 7주 이동평균선을 한 방향으로 교차한 지 10봉 이내에 반대 방향으로 교차하면 횡보장이다

    ## set price variables

    close_price = data.Close
    high_price = data.High
    open_price = data.Open
    low_price = data.Low

    ## 주간 이동평균선
    wma = close_price.rolling(7).mean()
    wma = wma.shift(1)
    data_merged = pd.merge(close_price.shift(1),close_price,left_index=True,right_index=True)
    data_merged = pd.merge(data_merged,wma,left_index=True, right_index=True)
    data_merged.rename(columns={'Close_x':'Open',"Close_y":'Close',"Close":'MA'},inplace=True)

    cross_date = []
    cross_date = data_merged[data_merged.MA < data_merged.Close]
    cross_date = cross_date[cross_date.MA > cross_date.Open]

    return cross_date

def _isBear(data):
    ## 주가가 7주 이동평균선을 상향 교차하면 상승장이다
    ## 주가가 7주 이동평균선을 하향 교차하면 하락장이다
    ## 주가가 7주 이동평균선을 한 방향으로 교차한 지 10봉 이내에 반대 방향으로 교차하면 횡보장이다

    ## set price variables

    close_price = data.Close
    high_price = data.High
    open_price = data.Open
    low_price = data.Low

    ## 12주 이동평균선
    wma = close_price.rolling(7).mean()
    wma = wma.shift(1)
    data_merged = pd.merge(close_price.shift(1),close_price,left_index=True,right_index=True)
    data_merged = pd.merge(data_merged,wma,left_index=True, right_index=True)
    data_merged.rename(columns={'Close_x':'Open',"Close_y":'Close',"Close":'MA'},inplace=True)

    cross_date = []
    cross_date = data_merged[data_merged.MA > data_merged.Close]
    cross_date = cross_date[cross_date.MA < cross_date.Open]

    return cross_date

bull_date = _isBull(raw_data)
bear_date = _isBear(raw_data)

tr_profit = 100.0
tr_profit2 = 100.0

profit_list =[]
profit_list2 =[]
kospi_list = []
kospi_list2 = []

### Bull 신호가 연속으로 중복되거나 Bear 신호가 연속으로 중복되는 경우 제거
def cleansing(bull_date,bear_date):
    bear_date_list = []
    bull_date_list = []

    for i in range(0, len(bull_date.index) - 1):
        axis_date = bull_date.index[i]
        next_date = bull_date.index[i + 1]

        end_date = bear_date[bear_date.index > axis_date].index[0]
        try:
            if bear_date_list[-1] != end_date:
                bear_date_list = bear_date_list+[end_date]
        except:
            if bear_date.index[0] < bull_date.index[0]:
                bear_date_list = [bear_date.index[0]] + [end_date]
            else:
                bear_date_list = [end_date]

    for i in range(0, len(bear_date.index) - 1):
        axis_date = bear_date.index[i]
        next_date = bear_date.index[i + 1]

        end_date = bull_date[bull_date.index > axis_date].index[0]

        try:
            if bull_date_list[-1] != end_date:
                bull_date_list = bull_date_list+[end_date]
        except:
            if bull_date.index[0] < bear_date.index[0]:
                bull_date_list = [bull_date.index[0]] + [end_date]
            else:
                bull_date_list = [end_date]

    return bull_date_list, bear_date_list

def profit_cal(data,other_data,dates,other_dates,key = 1):
    profit_list = []
    kospi_list = []
    for i in range(0,min(len(dates),len(other_dates))):
        start_price = float(data[data.index == dates[i]].MA)
        end_price = float(other_data[other_data.index > dates[i]].MA[0])

        if key == 1:
            profit = (end_price / start_price) - 1 -0.02 #### 실질적으로 MA 돌파시에 바로 포지션을 바꿀 수 없기 때문에 2% 손해 가정
        elif key == -1:
            profit = (start_price / end_price) -1  -0.02 #### 실질적으로 MA 돌파시에 바로 포지션을 바꿀 수 없기 때문에 2% 손해 가정

        profit_list = profit_list + [profit]
        kospi_list = kospi_list + [data.Close[i]]

    df = {'Kospi': kospi_list, 'profit': profit_list}
    df = pd.DataFrame(df, index=dates[0:min(len(dates),len(other_dates))])

    return df



##### 함수영역 끝 ########################################################


bull_dates, bear_dates = cleansing(bull_date,bear_date)
df1 = profit_cal(bull_date,bear_date,bull_dates,bear_dates,1)
df2 = profit_cal(bear_date,bull_date,bear_dates,bull_dates,-1)


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

print("MDD = ",round(df_mer.min()*100,2),"%")
print("Mean = ",round(df_mer.mean()*100,2),"%")
print("Hit Rataio = ",round(len(df_mer[df_mer>0])/len(df_mer),2))
ax = df_mer.plot()
ax.axhline(y=0)
# df_mer.plot.hist(bins=50)
df_profit.plot()
plt.show()
