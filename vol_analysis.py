import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

class isXberger:
    def __init__(self,data):
        self.data = data

    def x_times(self,times = 3):
        data = self.data

        start_price = data[0]

        # for i in data[1:]:
        #     if i/start_price > times:
        #         print(data.name)
        #         # plt.plot(data)
        #         # plt.show()
        #         break

        if data[-1]/start_price > 3:
            return data.name

class vol_analysis:
    def __init__(self,price_data,vol_data,equity_float):
        self.price_data = price_data
        self.vol_data = vol_data
        self.equity_float = equity_float

    def transact_vol_ratio(self):
        price_data = self.price_data
        vol_data = self.vol_data
        equity_float = self.equity_float

        trans_vol_ratio = pd.DataFrame(vol_data/(equity_float*1000000))
        trans_vol_df = trans_vol_ratio.sort_index(ascending=False)

        ful_vol_dates_index = []
        for i in range(0,len(trans_vol_df.index)):
            sums = 0
            for j in range(i,len(trans_vol_df)):
                sums = sums + trans_vol_df.values[j]
                if sums > 1:
                    ful_vol_dates_index = ful_vol_dates_index + [trans_vol_df.index[j]]
                    break

        print(len(ful_vol_dates_index))

        trans_vol_df = trans_vol_df.loc[trans_vol_df.index[0]:trans_vol_df.index[len(ful_vol_dates_index)-1]]
        print(len(trans_vol_ratio.index))

        trans_vol_df['ful_vol_dates'] = ful_vol_dates_index
        print(trans_vol_ratio.head())
        print(price_data['2019-07-12'])
        print(price_data['2018-12-06':'2019-07-12'])
        print(trans_vol_ratio['2018-12-06':'2019-07-12'])

        plt.subplot(2,1,1)
        plt.plot(trans_vol_ratio)

        plt.subplot(2,1,2)
        plt.plot(price_data)
        plt.show()



if __name__ == '__main__':
    data = pd.read_csv('kospi_daily_firmwise.csv',index_col='Dates',parse_dates=True)
    vol_data = pd.read_csv('kospi_daily_firmwise_volume.csv',index_col='Dates',parse_dates=True)
    eq_float_data = pd.read_csv('kospi_daily_firmwise_eqt_float.csv',index_col='Dates',parse_dates=True)

    cols = data.columns


    # b = vol_analysis(vol_data[cols[0]],eq_float_data[cols[0]])
    # b.transact_vol_ratio()



    # for i in data.columns:
    #     a = isXberger(data[i])
    #     names = a.x_times()
    #
    #     try:
    #         b = vol_analysis(data[names],vol_data[names],eq_float_data[names])
    #         print(names)
    #         b.transact_vol_ratio()
    #     except:
    #         pass


    b = vol_analysis(data['036570 KS Equity'],vol_data['036570 KS Equity'],eq_float_data['036570 KS Equity'])
    b.transact_vol_ratio()
