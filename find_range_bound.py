from scipy.signal import argrelextrema
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class range_bound_finder:
    def __init__(self,data,window = 26):
        self.data = data
        self.window = window

    def find_local_top(self):
        data = self.data
        max_data = argrelextrema(data.values,np.greater)
        # print(max_data[0])
        #
        # print(data.index[max_data[0]])
        # data.plot()
        # plt.show()
        return max_data[0]

    def clensing(self): ## 가까이 있는
        data = self.data
        local_maximal = self.find_local_top()
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

        if np.max(data.values) == np.max(local_max_list) and np.max(local_max_list)/np.min(local_max_list)-1 < 0.05:
            return 1



if __name__ == "__main__":
    data = pd.read_csv('kospi_data.csv',index_col='Date',parse_dates=True)
    data_fitted = data[['Close']].sort_index()

    for i in range(0,len(data_fitted.index)-26,6):
        a = range_bound_finder(data_fitted[i:i+26],window = 26)
        print(a.clensing())
