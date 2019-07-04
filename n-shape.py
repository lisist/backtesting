import pick_point as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('kospi_data.csv', index_col='Date', parse_dates=True)
data = data.sort_index()

for i in range(0,len(data.index)-26,13):
    a = pp.pickPoint(data[i:i+26], 3)
    peak_index = a.selectPeak()
    bottom_index = a.selectBottom()

    peak_date = data[i:i+26].index[peak_index]
    peak_price = data[i:i+26].Close.values[peak_index]

    bottom_date = data[i:i+26].index[bottom_index]
    bottom_price = data[i:i+26].Close.values[bottom_index]

    for j in range(0,len(peak_index)-1):
        if i > 52:
            maximum = np.max(data[i+peak_index[j] - 52:i+peak_index[j]].High)
            if peak_price[j] >= maximum:
                print(peak_date[j],peak_price[j])
                next_bottom_date = bottom_date[bottom_date > peak_date[j]][0]
                next_bottom_price = bottom_price[bottom_date>peak_date[j]][0]

                if peak_price[j] > next_bottom_price and peak_price[j+1] > peak_price[j]:
                    print(peak_date[j+1],peak_price[j+1])

                    # print(data[data.index > peak_date[j] & data.index < data[i+26].index].Close > peak_price[j])



                    data[i:i + 26].Close.plot()
                    plt.scatter(x=peak_date[j], y=peak_price[j])
                    plt.scatter(x=next_bottom_date, y=next_bottom_price)
                    plt.scatter(x=peak_date[j + 1], y=peak_price[j + 1])
                    plt.show()


        else:
            pass

