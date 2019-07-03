import pick_point as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('kospi_data.csv', index_col='Date', parse_dates=True)
data = data.sort_index()

for i in range(0,len(data.index)-26,13):
    a = pp.pickPoint(data[i:i+26], 3)
    peak_index = a.selectPeak()
    peak_date = data[i:i+26].index[peak_index]
    peak_price = data[i:i+26].High.values[peak_index]


    for j in range(0,len(peak_index)):
        if i > 52:
            maximum = np.max(data[i+peak_index[j] - 52:i+peak_index[j]].High)
            if peak_price[j] >= maximum:
                print(peak_date[j],peak_price[j])

            a.plotPeaks()
            a.plotBottoms()

            plt.show()

        else:
            pass
