import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


### class short_and
### 지난 몇 일간(예를 들어 4주) 변동성을 넘어 장이 하락할 때
### 다음날 시가 기준으로 진입.
### 이후 평균적으로 최대로 얼마까지 빠진 이후 반등하는 지 확인

class short_and:
    def __init__(self,data):
        self.data = data

    def data_filtered(self):
        data = self.data
        data = data.dropna()
        data = data[['Open','High','Low','Close','Volume']]

        data.sort_index(inplace=True)

        return data


    def average_vol(self):
        ### 평균 n일 동안의 변동성 확인 (ATR 이용)
        ### 이를 이용하여

        data = self.data_filtered()

        data['dif1'] = data.High - data.Low
        data['dif2'] = np.abs(data.High - data.shift(1).Close)
        data['dif3'] = np.abs(data.Low - data.shift(1).Close)
        data['TR'] = data[['dif1','dif2','dif3']].max(axis=1)
        data['ATR'] = data.TR.rolling(window=20).mean()

        data.dropna(inplace=True)

        data = data[['Open','High','Low','Close','Volume','ATR']]

        return data

    def entry_porint(self):
        data = self.average_vol()

        data['daily_diff'] = data.Close-data.Close.shift(1)
        entry_point = data[data.daily_diff < -data.ATR].index

        return entry_point

    def calcul(self):
        data = self.average_vol()
        entry_point = self.entry_porint()

        profit_list =[]
        entry_point_list = []
        for i in entry_point:
            data2 = data[data.index >= i]
            data2 = data2[(data2.index - i).days < 120]

            entry_price = data2.Open[1]
            data2['price_chg'] = data2.Low - entry_price
            data2 = data2.iloc[2:]

            for j in range(0,len(data2.index)):
                if data2['price_chg'][j] > 0:
                    data2 = data2.iloc[:j]
                    # print(data2)
                    break

            try :
                min_chg = np.min(data2['price_chg'])
                min_profit = min_chg/entry_price

                profit_list = profit_list + [min_profit]
                entry_point_list = entry_point_list + [i]

            except:
                pass

        print(profit_list)
        print(np.nansum(profit_list)/len(entry_point))

        df = {'date':entry_point_list,'profit':profit_list}
        df = pd.DataFrame(df)
        df.to_csv('short_and_result.csv')
        plt.bar(x=range(0,len(entry_point)),height=profit_list)
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('kospi_daily.csv', index_col='Date', parse_dates=True)

    # a = short_and(data[:-2500])
    a = short_and(data)
    # a.data_filtered()
    a.calcul()
