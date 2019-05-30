## 전략 001

## 거래차트 : 주봉

## Case1 : Long 전략
## 진입 전략 : 7주 이평선을 장대 양봉으로 돌파한 순간 진입
## 청산 전략 : 시장이 장대 양봉 돌파 순간의 최저점을 '최저 저항선'으로 지정하고 가격이 최저 저항선 아래로 내려가지 않는 상황에서는
##              Holding period간 보유

## Case2 : Short 전략
## 진입 전략 : 7주 이평선을 장대 음봉으로 하향 돌파한 순간 진입
## 청산 전략 : 시장이 장대 음봉 돌파 순간의 최고점을 '최저 저항선'으로 지정하고 가격이 최저 저항선 위로 올라가지 않는 상황에서는
##              Holding period간 보유


## 향후 보완점 : False signal 발생 이후 청산할 때 수익률을 -1% 일괄 적용했지만 실제로는 청산하는 주의 Close price를 이용하여 계산해야 할 듯
#####################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def data_merge(data, window=7):
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


def charting(df1,data,profit_list,window,holding_period):
    print("Number of Opportunities = ",len(profit_list))
    # print("Number of False signal",(profit_list))
    print("MA weeks = ", window)
    print("hodling period =", holding_period, "months")
    print("Total Return = ", df1.profit[-1])
    print("Benchmark Return = ", df1.Kospi[-1])
    print("MDD  = ", round(min(a.profit()) * 100, 2), "%")
    print("Hit Rataio (if no false signal) = ", round(len(profit_list[profit_list > 0]) / len(profit_list), 2))
    print("Expected return (if no false signal) = ", round(profit_list.mean()*100,2),"%")

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(df1.index, df1.profit, 'y-')
    plt.title("Strategy Total Return")
    plt.ylabel("(Start index = 100)")
    # plt.yscale('log')
    print(ax1)

    ax2 = plt.subplot(3, 1, 2)
    plt.plot(data.index,data.Close,'r-')
    plt.title("Benchmark Return")
    plt.xlabel('Dates')
    print(ax2)

    ax3 = plt.subplot(3,1,3)
    plt.hist(profit_list,bins=60)
    plt.title("Return histogram")
    plt.xlabel('Dates')
    plt.ylabel("unit")
    print(ax3)
    #
    plt.tight_layout()
    plt.show()


class Strategy:
    def __init__(self,data,holding_period=3, cost=0.05, key=1):
        self.data = data
        self.key = key  ## if key == 1 : bull market signal, if key == -1 : bear markey signal
        self.holding_period = holding_period
        self.cost = cost
        self.location = []

    def pilar(self):
        ### 장대음봉이고 7주 이평선이 장대 음봉 사이를 지나는지 확인하는 함수 ####
        data = self.data[self.data.index<self.data.index[-self.holding_period*4]]
        data_pilar = []
        if self.key == 1:
            for i in range(0,len(data.index)):
                if data.Close[i]-data.Open[i] > (data.High[i]-data.Low[i])*0.5:
                    if  data.MA[i] > data.Low[i] and data.MA[i] < data.Close[i]:
                        data_pilar = data_pilar + [data.index[i]]
        if self.key == -1:
            for i in range(0,len(data.index)):
                if data.Open[i]-data.Close[i] > (data.High[i]-data.Low[i])*0.5:
                    if  data.MA[i] < data.High[i] and data.MA[i] > data.Close[i]:
                        data_pilar = data_pilar + [data.index[i]]

        return data_pilar

    def least_resistance(self):
        ### 최소 저항선 설정 #####
        ### 일반적으로 장대 양봉이 7주 이평선에 도달한 이후 양봉의 최저점을 최저 저항선으로 설정
        data_pilar = self.pilar()
        lr_list = []

        if self.key == 1:
            for i in range(0, len(data_pilar)):
                lr = self.data[self.data.index == data_pilar[i]].Low
                lr_list = lr_list + [float(lr)]

        if self.key == -1:
            for i in range(0,len(data_pilar)):
                lr = self.data[self.data.index==data_pilar[i]].High
                lr_list = lr_list + [float(lr)]

        return lr_list

    def searchFalseSignal(self):
        ### Moving Average 선을 돌파한 이후 1% 이상 상승하였지만 Close 가격이 다시 Moving Average 선 아래로 내려가는 경우를 False Signal로 정의
        ### 1% 선까지 올라갔다가 다시 MA 선을 하향 돌파하는 순간 포지션 정리. 이 경우 Profit에서 -1% 로 기록
        falsesignal =[]

        if self.key == 1:
            for i in range(0,len(self.data)):
                if self.data.Close[i] < self.data.MA[i] and self.data.Open[i] < self.data.MA[i] and self.data.High[i] > self.data.MA[i] :
                    if self.data.High[i]/self.data.MA[i] > 1.01:
                        falsesignal = falsesignal+[self.data.index[i]]

        if self.key == -1:
            for i in range(0,len(self.data)):
                if self.data.Close[i] > self.data.MA[i] and self.data.Open[i] > self.data.MA[i] and self.data.Low[i] < self.data.MA[i] :
                    if self.data.Low[i]/self.data.MA[i] < 0.99:
                        falsesignal = falsesignal+[self.data.index[i]]

        return falsesignal

    def profit(self):
        pilar_date = self.pilar()
        lr_data = self.least_resistance()
        false_date = self.searchFalseSignal()
        print(pilar_date)

        profit_list = np.repeat(0.0, len(pilar_date))
        time_frame = 4 * self.holding_period  ### short의 경우 Long보다 짧게
        self.location = np.repeat(0, len(pilar_date))

        for i in range(0, len(pilar_date)):
            reach_leastResistance = False

            if self.key == 1:
                ##### 최소 저항선을 돌파하면 그 순간 loss taking. 그렇지 않으면 holding period까지 holding한 이후 profit/loss taking
                for j in range(0, time_frame):
                    self.location[i] = len(self.data[self.data.index <= pilar_date[i]].index)
                    # print(self.location)
                    if self.data.iloc[self.location[i] + j].Low < lr_data[i]:
                        profit = ((lr_data[i] * (1 - self.cost)) / (
                                self.data[self.data.index == pilar_date[i]].MA * (1 + self.cost + 0.005)) - 1)
                        profit_list[i] = float(profit)
                        reach_leastResistance = True

                if reach_leastResistance is False:
                    profit = ((self.data.iloc[self.location[i] + time_frame - 1].Close ) / (
                                self.data.iloc[self.location[i] - 1].MA * (1 + self.cost+0.005)) - 1)
                    profit_list[i] = float(profit)

            if self.key == -1:
                ##### 최소 저항선을 돌파하면 그 순간 loss taking. 그렇지 않으면 holding period까지 holding한 이후 profit/loss taking
                for j in range(0, time_frame):
                    self.location[i] = len(self.data[self.data.index <= pilar_date[i]].index)
                    if self.data.iloc[self.location[i] + j].High > lr_data[i]:
                        profit = -((lr_data[i] * (1 + self.cost)) / (
                                self.data[self.data.index == pilar_date[i]].MA * (1 - self.cost - 0.005)) - 1)
                        profit_list[i] = float(profit)
                        reach_leastResistance = True

                if reach_leastResistance is False:
                    profit = -((self.data.iloc[self.location[i] + time_frame - 1].Close ) / (
                                self.data.iloc[self.location[i] - 1].MA * (1 - self.cost-0.005)) - 1)
                    profit_list[i] = float(profit)



        profit_list = list(profit_list)

        return profit_list

    def total_profit(self):

        tr_profit = 100.0
        tr_profit_list = []
        exit_date_list = []

        profit_list = self.profit()
        false_date = self.searchFalseSignal()
        pilar_date = self.pilar()

        kk = 0
        le_kk = len(profit_list)

        for i in range(0, len(false_date)):
            for j in range(0, le_kk + kk):
                if (false_date[i] - pilar_date[j]).days < 0:
                    profit_list.insert(j, -0.01)
                    pilar_date.insert(j, false_date[i])
                    # print(i,"",j)
                    kk += 1
                    break

        for i in profit_list:
            tr_profit = tr_profit * (1 + i)
            tr_profit_list = tr_profit_list + [tr_profit]

        kk = 0
        time_frame = 4 * self.holding_period
        for i in self.location:
            exit_date_list = exit_date_list + [self.data.iloc[i + time_frame - 1].name]
        for i in range(0, len(false_date)):
            for j in range(0, le_kk + kk):
                if (false_date[i] - exit_date_list[j]).days < 0:
                    exit_date_list.insert(j, false_date[i])
                    kk += 1
                    break

        benchmark_price_list =[]
        for i in exit_date_list:
            benchmark_price = float(self.data[self.data.index == i].Close)
            benchmark_price_list = benchmark_price_list + [benchmark_price]

        benchmark_price_list = [x/benchmark_price_list[0]*100 for x in benchmark_price_list]

        df = {"Kospi":benchmark_price_list,"profit":tr_profit_list}
        df = pd.DataFrame(df,index=exit_date_list)

        return df


if __name__ == '__main__':
    raw_data = pd.read_csv('kospi_data.csv', index_col='Date', parse_dates=True)
    raw_data.sort_index(inplace=True)

    window = 7  ## MA 계산할 주기 기본 7주
    holding_period = 3
    cost = 0.005

    data = data_merge(raw_data)
    data = data[data.index>data.index[-200]]

    a = Strategy(data,holding_period,cost,key=1)
    b = Strategy(data,holding_period=1,cost=0.005,key =-1)

    profit_list = np.array(a.profit())
    df1 = a.total_profit()
    charting(df1,data,profit_list,window,a.holding_period)

    profit_list2 = np.array(b.profit())
    df2 = b.total_profit()
    charting(df2,data,profit_list2,window,b.holding_period)
