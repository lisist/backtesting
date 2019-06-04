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

    def calCorrel(self):
        data = self.data
        target_data = data['SPX'][-128:]

        le = len(data)
        cor_list = []
        date_list =[]
        for i in range(0, le-256):
            com_data = data['SPX'][i:i + 128]
            cor_list = cor_list + [np.corrcoef(target_data.values, com_data.values)[1, 0]]
            date_list = date_list + [data.index[i]]

        df = {'date':date_list,'cor':cor_list}
        df = pd.DataFrame(df)

        return df

    def signalDates(self):
        data = self.data
        df = self.calCorrel()
        df_test = df.sort_values(by=['cor'], ascending=False)

        signal_date_list = []

        for i in range(0,5):
            signal_date = df_test.iloc[0].date
            df_test = df_test.drop(df_test[abs((df_test['date'] - signal_date).dt.days) < 128].index, axis=0)
            signal_date_list = signal_date_list + [signal_date]

        return signal_date_list

    def decideLongShort(self):
        data = self.data
        dates = self.signalDates()

        profit_list = []

        for i in range(0,5):
            current_price = data[data.index == dates[i]].values[0]
            future_price = data[(data.index - dates[i]).days < 90].values[-1]

            profit = float(future_price / current_price - 1)
            profit_list = profit_list + [profit]

        average_return = np.mean(np.array(profit_list))

        if average_return > 0.1:
            return 1
        else:
            return 0


if __name__=='__main__':

    data = pd.read_csv('spx_all_time.csv', index_col='date', parse_dates=True)
    monthly_data = pd.read_csv('spx_monthly',index_col='date',parse_dates=True)
    data.sort_index(inplace=True)
    monthly_data.sort_index(inplace=True)

    for i in range(0,2000):
        data_selected = data[:-2000+i]
        a = Strategy3(data_selected)

        print(data_selected.index[-1])
        print(a.decideLongShort())
