from scipy.signal import argrelextrema
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class range_bound_finder:
    def __init__(self,data,window = 26):
        self.data = data
        self.window = window

    def find_local_top(self,type="greater"):
        data = self.data
        if type=="greater":
            return argrelextrema(data.values,np.greater)[0]
        if type=="less":
            return argrelextrema(data.values, np.less)[0]


    def clensing_max(self): ## 가까이 있는
        data = self.data
        local_maximal = self.find_local_top(type='greater')
        window = self.window

        tab = int(window/8)

        local_max_list = []

        for i in range(0,len(local_maximal)):
            if i != 0 and local_maximal[i] - local_maximal[i-1] > tab:
                local_max_list = local_max_list + [data.Close[local_maximal[i]]]
            elif i != 0 and local_maximal[i] - local_maximal[i-1] <= tab:
                if data.Close[local_maximal[i]] > data.Close[local_maximal[i-1]]:
                    del local_max_list[-1]
                    local_max_list = local_max_list + [data.Close[local_maximal[i]]]
            elif i == 0:
                local_max_list = local_max_list + [data.Close[local_maximal[i]]]
            else:
                pass

        return local_max_list


    def clensing_min(self):  ## 가까이 있는
        data = self.data
        local_minimal = self.find_local_top(type='less')
        window = self.window

        tab = int(window / 8)

        local_min_list = []

        for i in range(0, len(local_minimal)):
            if i != 0 and local_minimal[i] - local_minimal[i - 1] > tab:
                local_min_list = local_min_list + [data.Close[local_minimal[i]]]
            elif i != 0 and local_minimal[i] - local_minimal[i - 1] <= tab:
                if data.Close[local_minimal[i]] < data.Close[local_minimal[i - 1]]:
                    del local_min_list[-1]
                    local_min_list = local_min_list + [data.Close[local_minimal[i]]]
            elif i == 0:
                local_min_list = local_min_list + [data.Close[local_minimal[i]]]
            else:
                pass

        return local_min_list


    def slope_judgement(self):  ## 가까이 있는
        data = self.data
        local_max_list = self.clensing_max()
        local_min_list = self.clensing_min()

        x = range(0,len(local_max_list))
        max_slope = linregress(x,local_max_list)[0]
        max_slope = max_slope/np.mean(local_max_list)

        x = range(0,len(local_min_list))
        min_slope = linregress(x,local_min_list)[0]
        min_slope = min_slope/np.mean(local_min_list)


        if np.abs(max_slope + min_slope) < 0.03 and np.max(data.Close) < np.max(local_max_list) * 1.03 and np.min(data.Close) > np.min(local_min_list) *0.97:
            return 1
        else:
            return 0


def in_range_ratio(data,high,low):
    upper = (data.Close <= high)*1
    lower = (data.Close >= low)*1

    # in_range = lower*upper
    in_range = lower

    inRangeRatio = np.sum(in_range)/len(data.Close)

    return inRangeRatio

def strategy(data,high,low):
    ### low 선을 하향 돌파할 때 buy -> low 선보다 3% 이하로 떨어질 때 loss cut
    ### low 선을 하향 돌파한 이후 다시 상향 돌파할 때 추가 buy
    ### low 선 위에 있을 때에는 hold

    b_signal = 0
    ret = 0
    for i in range(0,len(data.index)):
        if  data.Close[i] < low  and data.Close[i] > low * 0.97 and b_signal == 0:
            buy_price = data.Close[i]
            b_signal = 1
        elif data.Close[i] < low * 0.97 and b_signal != 0:
            ret = (1+ret) *(data.Close[i]/buy_price)-1
            b_signal = 0
        elif i >= 1 and data.Close[i-1] < low and data.Close[i] > low and b_signal != 0:
            buy_price2 = data.Close[i]
            b_signal = 2
        else:
            pass


    if b_signal == 1:
        ret = (1+ret) * (data.Close[-1]/buy_price)-1
    elif b_signal == 2:
        ret  = (1+ret) * (data.Close[-1]/buy_price)*(data.Close[-1]/buy_price2)-1

    try:
        # print(buy_price," ",buy_price2," ",data.Close[-1])
        return ret
    except:
        pass

    # data.plot()
    # plt.axhline(y=high)
    # plt.axhline(y=low)
    # plt.show()


if __name__ == "__main__":
    data = pd.read_csv('kospi_data.csv',index_col='Date',parse_dates=True)
    data_fitted = data[['Close']].sort_index()

    inRangeRatioList = []

    for i in range(0,len(data_fitted.index)-26,6):
        a = range_bound_finder(data_fitted[i:i+26],window = 26)

        if a.slope_judgement() == 1:
            min_list = a.clensing_min()
            max_list = a.clensing_max()

            inRangeRatio = in_range_ratio(data_fitted[i+26:i+52],np.max(max_list),np.min(min_list))
            inRangeRatioList = inRangeRatioList + [inRangeRatio]

            st1 = strategy(data_fitted[i+26:i+52],np.max(max_list),np.min(min_list))
            print(st1)
