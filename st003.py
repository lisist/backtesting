## 전략 003

## 거래차트 : 주봉

## 기본 아이디어
## 과거와 비슷한 사례를 찾아 (Correlation)으로, 비슷한 사례 5건의 평균이 수익권이라면 진입
## 만약 이후 비슷한 사례에서 제시한 방향보다 아래로 내려간다면 청산

## Case1 : Long 전략
## 진입 전략 : RSI(14)가 30이하일 때 진입
## 청산 전략 : RSI(14)가 30이하에서 돌아설 때 최저점을 최저 저항선으로 지정하고 가격이 최저 저항선 아래로 내려가지 않는 상황에서는
##              Holding period간 보유


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Strategy3:
    def __init__(self,data):
        self.data = data

    def daily_return(self):
        data = self.data
        d_return = data / data.shift(1) - 1

        return d_return[1:]

    def calCorrel(self):    ### 128일 이전 Correlation을 계산합니다
        data = self.daily_return()
        target_data = data['SPX'][-128:]

        le = len(data)
        cor_list = []
        date_list =[]
        for i in range(0, le-256):
            com_data = data['SPX'][i:i + 128]
            cor_list = cor_list + [np.corrcoef(target_data.values, com_data.values)[1, 0]]
            date_list = date_list + [data.index[i+127]]

        df = {'date':date_list,'cor':cor_list}
        df = pd.DataFrame(df)

        return df

    def signalDates(self):   ### Correaltion이 가장 높은 5개 시기의 시작 일자를 뽑습니다.
        # data = self.data
        df = self.calCorrel()
        df_test = df.sort_values(by=['cor'], ascending=False)

        signal_date_list = []

        for i in range(0,10):
            signal_date = df_test.iloc[0].date
            df_test = df_test.drop(df_test[abs((df_test['date'] - signal_date).dt.days) < 128].index, axis=0)
            signal_date_list = signal_date_list + [signal_date]

        return signal_date_list

    def decideLongShort(self):   ### Long할지 여부를 결정합니다. Correlation이 높은 5개 시기의 향후 3개월간 수익률이 hurdle_rate 수준 이상이면 Long
        data = self.data
        dates = self.signalDates()
        hurdle_return = 0.03

        profit_list = []

        for i in range(0,2):
            current_price = data[data.index == dates[i]].values[0]
            future_price = data[(data.index - dates[i]).days < 31].values[-1]

            profit = float(future_price / current_price - 1)
            profit_list = profit_list + [profit]

        average_return = np.mean(np.array(profit_list))

        # if average_return > hurdle_return:
        #     return 1
        # else:
        #     return 0

        # if sum([x < 0 for x in profit_list]) >= 7:
        #     return 1
        if average_return < -hurdle_return:
            return 1
        else:
            return 0


if __name__=='__main__':

    data = pd.read_csv('spx_all_time.csv', index_col='date', parse_dates=True)

    monthly_data = data.resample('M').last()  ## 월별 데이터로 정리
    monthly_data = monthly_data[-230:]

    a_list = []
    threeMReturn_list = []
    dates = []


    for i in range(0,len(monthly_data.index)-5):
        data_selected = data[data.index <= monthly_data.index[i]]
        dates = dates + [monthly_data.index[i]]
        a = Strategy3(data_selected)

        a_list = a_list + [a.decideLongShort()]
        threeMReturn_list = threeMReturn_list +  [monthly_data.iloc[i+1].SPX / monthly_data.iloc[i].SPX - 1]

        print(monthly_data.index[i])
        print(a_list[-1])
        print(threeMReturn_list[-1])

        df = {'Date': dates, 'Signal': a_list, 'Return': threeMReturn_list}
        df = pd.DataFrame(df)
        df.to_csv('result.csv')
