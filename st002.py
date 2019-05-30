## 전략 002

## 거래차트 : 주봉
## RSI와 ATR의 조합으로 최적의 진입 및 청산 전략 찾기

## Case1 : Long 전략
## 진입 전략 : RSI(14)가 30이하일 때 진입
## 청산 전략 : RSI(14)가 30이하에서 돌아설 때 최저점을 최저 저항선으로 지정하고 가격이 최저 저항선 아래로 내려가지 않는 상황에서는
##              Holding period간 보유

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def merging(data):
    colNames = data.columns
    df = pd.merge(data[colNames[0:4]],data[colNames[-1]],left_index=True, right_index=True)
    df.columns = ['Open','High','Low','Close','RSI(14)']

    return df


class Strategy2:
    def __init__(self,data, target=30, holding_period=3):
        self.data = data
        self.target = target
        self.holding_period = holding_period
        self.key = 1
        self.location = []
        self.cost = 0.005

    def signal_date(self):
        data = self.data
        target = self.target

        signalDate = data[data['RSI(14)'] < target].index

        return signalDate

    def least_resistance(self):
        ### 최소 저항선 설정 #####
        ### 일반적으로 장대 양봉이 7주 이평선에 도달한 이후 양봉의 최저점을 최저 저항선으로 설정
        date = self.signal_date()
        lr_list = []

        if self.key == 1:
            for i in range(0, len(date)):
                lr = data[data.index == date[i]].Low
                lr_list = lr_list + [float(lr)]

        if self.key == -1:
            for i in range(0,len(data_pilar)):
                lr = data[self.data.index== date[i]].High
                lr_list = lr_list + [float(lr)]

        return lr_list


    def profit(self):
        signalDate = self.signal_date()
        data = self.data
        lr_data = self.least_resistance()

        profit_list = np.repeat(0.0, len(signalDate))
        location = np.repeat(0, len(signalDate))
        time_frame = 4 * self.holding_period  ### short의 경우 Long보다 짧게

        for i in range(0, len(signalDate)):
            reach_leastResistance = False

            if self.key == 1:
                ##### 최소 저항선을 돌파하면 그 순간 loss taking. 그렇지 않으면 holding period까지 holding한 이후 profit/loss taking
                for j in range(0, time_frame):
                    location[i] = len(data[data.index <= signalDate[i]].index)

                    if data.iloc[location[i] + j].Low < lr_data[i]:
                        profit = (lr_data[i] * (1 - self.cost)) / (data[data.index == signalDate[i]].Close)  - 1
                        profit_list[i] = float(profit)
                        reach_leastResistance = True

                if reach_leastResistance is False:
                    profit = (data.iloc[location[i] + time_frame - 1].Close ) / (
                                data.iloc[location[i] - 1].Close)- 1
                    profit_list[i] = float(profit)

            # if self.key == -1:
            #     ##### 최소 저항선을 돌파하면 그 순간 loss taking. 그렇지 않으면 holding period까지 holding한 이후 profit/loss taking
            #     for j in range(0, time_frame):
            #         self.location[i] = len(self.data[self.data.index <= pilar_date[i]].index)
            #         if self.data.iloc[self.location[i] + j].High > lr_data[i]:
            #             profit = -((lr_data[i] * (1 + self.cost)) / (
            #                     self.data[self.data.index == pilar_date[i]].MA * (1 - self.cost - 0.005)) - 1)
            #             profit_list[i] = float(profit)
            #             reach_leastResistance = True
            #
            #     if reach_leastResistance is False:
            #         profit = -((self.data.iloc[self.location[i] + time_frame - 1].Close ) / (
            #                     self.data.iloc[self.location[i] - 1].MA * (1 - self.cost-0.005)) - 1)
            #         profit_list[i] = float(profit)


        self.location = location
        profit_list = list(profit_list)

        return profit_list

    def total_profit(self):
        profit_list = self.profit()
        date = self.signal_date()
        tr_profit = 100
        tr_profit_list = []

        for i in profit_list:
            tr_profit = tr_profit * (1+i)
            tr_profit_list = tr_profit_list + [tr_profit]

        print(tr_profit_list)


if __name__ == '__main__':
    raw_data = pd.read_csv('kospi_data.csv', index_col='Date', parse_dates=True)
    raw_data.sort_index(inplace=True)

    data = merging(raw_data)

    target_RSI = 35

    a = Strategy2(data,target_RSI)
    # print(a.signal_date())
    # print(a.profit())

    df = {'date':a.signal_date(),'profit':a.profit()}
    df = pd.DataFrame(df)
    print(df)

    plt.plot(df.date,df.profit)
    plt.show()

    print(df)
    a.total_profit()

    # window = 7  ## MA 계산할 주기 기본 7주
    # holding_period = 3
    # cost = 0.005
