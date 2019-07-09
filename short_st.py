import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

### class short_and
### 지난 몇 일간(예를 들어 4주)의 변동성을 넘어 장이 하락할 때
### 다음날 시가 기준으로 진입
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

    def entry_point(self):
        data = self.average_vol()

        data['daily_diff'] = data.Close-data.Close.shift(1)
        entry_point = data[data.daily_diff < -data.ATR].index

        return entry_point

    def calCul(self):
        data = self.average_vol()
        entry_point = self.entry_point()
        divider = 5

        profit_list =[]
        entry_price_list = []
        exit_date = []
        exit_price_list = []

        for i in entry_point:

            data2 = data[data.index >= i]
            data2 = data2[(data2.index - i).days < 120]

            if len(data2.index) > 1:
                entry_price = data2.Open[1]     ## entry 신호 이후 바로 다음날 시초가로 진입
                entry_price_list = entry_price_list + [entry_price]
            else:
                break

            data2['price_chg'] = data2.Low - entry_price
            data2 = data2.iloc[1:]

            for j in range(0,len(data2.index)):
                if data2['price_chg'][j] > 0:
                    if j <= len(data2.index)-50:
                        data2 = data2.iloc[:j+50]
                        break_date = data2.index[j]
                        break
                    else:
                        data2 = data2
                        break_date = data2.index[j]
                        break

            ### exit 기준가 산정

            exit_price_chg = []

            exit_price_chg.append(data2.ATR[0]/divider + data2.price_chg[0])

            for j in range(1,len(data2.index)):
                exit_price_chg.append(exit_price_chg[j-1])
                if data2.ATR[j]/divider + data2.price_chg[j] < exit_price_chg[j] :
                    exit_price_chg[j]=(data2.ATR[j]/divider + data2.price_chg[j])
                else:
                    pass

                if data2.price_chg[j] > exit_price_chg[j]:
                    exit_date = exit_date + [data2.index[j]]
                    exit_price_list = exit_price_list + [exit_price_chg[j]]
                    break
                elif j == (len(data2.index) - 1):
                    exit_date = exit_date + [data2.index[j]]
                    exit_price_list = exit_price_list + [exit_price_chg[j]]

            if len(data2.index) == 1:
                exit_date = exit_date + [data2.index[0]]
                exit_price_list = exit_price_list + [data2.Close[0]-data2.Open[0]]



        ### Profit 계산
        profit_list = -np.array(exit_price_list)/np.array(entry_price_list)
        profit_list = list(np.array(profit_list)-0.003)   ### 수수료 등을 감안하여 평균 0.3% 비용 반영

        # print(profit_list)

        # df = {'date':entry_point,'profit':profit_list}
        # df = pd.DataFrame(df)
        # df.to_csv('short_and_result.csv')
        # plt.bar(x=range(0,len(entry_point)),height=profit_list)
        # plt.show()
        #
        # plt.hist(profit_list,bins=24)
        # plt.show()
        return profit_list


    def total_stat(self):
        profit_list = self.calCul()

        if len(profit_list) > 0:
            hit_ratio = sum([x>0 for x in profit_list])/len(profit_list)
            expected_return =  np.mean(profit_list)

            return hit_ratio, expected_return
        else:
            return np.nan, np.nan





if __name__ == '__main__':
    data = pd.read_csv('kospi_daily.csv', index_col='Date', parse_dates=True)
    key = np.random.randint(0,len(data.index)-1000,50)

    hit_ratio_list =[]
    expected_return_list = []

    for i in key:
        a = short_and(data[i:i + 1000])
        # hit_ratio, expected_return =a.total_stat()
        hit_ratio, expected_return = a.total_stat()
        print('hit_ratio : ', hit_ratio)
        print('expected_raturn : ', expected_return)

        hit_ratio_list = hit_ratio_list + [hit_ratio]
        expected_return_list = expected_return_list + [expected_return]


    print(np.nanmean(hit_ratio_list))
    print(np.nanmean(expected_return_list))

    # a = short_and(data[:-len(data.index)+300])
    # b = short_and(data[1000:2000])
    # c = short_and(data[2000:3000])
    # a.exit_date()
    # b.exit_date()
    # c.exit_date()

