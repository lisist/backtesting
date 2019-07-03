import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from scipy.stats import linregress


class pickPoint:


    def __init__(self,data,tab=3):
        self.data = data
        self.tab = tab



    def dataFit(self):
        ### 데이터를 정리합니다

        data = self.data
        data = data[['Open','High','Low','Close','Volume']]
        data = data.sort_index()

        return data


    def localPeak(self,type="greater"):
        ### 지역 극점을 모두 표시합니다

        data = self.dataFit()

        if type=="greater":
            data = data['High']
            return argrelextrema(data.values,np.greater)[0]

        if type=="less":
            data = data['Low']
            return argrelextrema(data.values, np.less)[0]


    def selectPeak(self):
        ### 지역 고점을 적당한 간격으로 선택합니다

        data = self.dataFit()
        localpeaks = self.localPeak(type='greater')

        tab = self.tab

        peak_list =[]
        for i in range(0,len(localpeaks)):
            if i != 0 and localpeaks[i] - peak_list[-1] > tab:
                peak_list = peak_list + [localpeaks[i]]
            elif i != 0 and localpeaks[i] - peak_list[-1] <= tab:
                if data.High[localpeaks[i]] > data.High[peak_list[-1]]:
                    del peak_list[-1]
                    peak_list = peak_list + [localpeaks[i]]
                elif data.High[localpeaks[i]] < data.High[peak_list[-1]] and localpeaks[i] - peak_list[-1] > tab * 3:
                    peak_list = peak_list + [localpeaks[i]]

            elif i == 0:
                peak_list = peak_list + [localpeaks[i]]
            else:
                pass



        return peak_list


    def selectBottom(self):
        ### 지역 저점을 적당한 간격으로 선택합니다

        data = self.dataFit()
        localpeaks = self.localPeak(type='less')

        tab = self.tab

        bottom_list =[]

        for i in range(0,len(localpeaks)):
            if i != 0 and localpeaks[i] - bottom_list[-1] > tab:
                bottom_list = bottom_list + [localpeaks[i]]
            elif i != 0 and localpeaks[i] - bottom_list[-1] <= tab:
                if data.Low[localpeaks[i]] < data.Low[bottom_list[-1]]:
                    del bottom_list[-1]
                    bottom_list = bottom_list + [localpeaks[i]]
                elif data.Low[localpeaks[i]] > data.Low[bottom_list[-1]] and localpeaks[i] - bottom_list[-1] > tab * 3:
                    bottom_list = bottom_list + [localpeaks[i]]

            elif i == 0:
                bottom_list = bottom_list + [localpeaks[i]]
            else:
                pass

        return bottom_list


    def plotPeaks(self):
        ## 선택된 고점들의 위치를 Plot합니다

        peak_list = self.selectPeak()

        x = self.dataFit().High[peak_list].index
        y = self.dataFit().High[peak_list].values

        plt.scatter(x,y, marker='s')
        plt.plot(self.dataFit().High)



    def plotBottoms(self):
        ## 선택된 저점들의 위치를 Plot합니다

        bottom_list = self.selectBottom()

        x = self.dataFit().Low[bottom_list].index
        y = self.dataFit().Low[bottom_list].values

        plt.scatter(x, y)
        self.dataFit()['Low'].plot()
