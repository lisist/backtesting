import pick_point as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('kospi_data.csv', index_col='Date', parse_dates=True)
data = data.sort_index()




for i in range(0,len(data.index)-26):
    data_selected = data[i:i+26]

    a = pp.pickPoint(data_selected, 3)

    peak_index = a.selectPeak()
    bottom_index = a.selectBottom()

    peak_date = data[i:i+26].index[peak_index]
    peak_price = data[i:i+26].High.values[peak_index]

    bottom_date = data[i:i+26].index[bottom_index]
    bottom_price = data[i:i+26].Low.values[bottom_index]

    peak_price = peak_price.tolist()
    max_price = np.max(peak_price)


    if data_selected.High[-1] >= max_price and len(peak_date) > 0:
        max_peak_date = peak_date[peak_price.index(max_price)]

        selected_bottom_date = bottom_date[bottom_date > max_peak_date]
        selected_bottom_price = bottom_price[bottom_date > max_peak_date]

        try:
            if max_price * 0.95 > np.min(selected_bottom_price) and len(selected_bottom_date) > 0:
                buy_date = data_selected.index[-1]
                print(buy_date)


                data_selected.Close.plot()
                data_selected.High.plot()
                data_selected.Low.plot()
                plt.scatter(x=peak_date, y=peak_price)
                plt.scatter(x=bottom_date, y=bottom_price)
                plt.show()

        except:
            pass
