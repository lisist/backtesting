import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class st0901:
    def __init__(self,data):
        self.data = data
        self.entry_way = int(np.random.randint(0,2,1)) # 0 long, 1 short


    def dataFit(self):
        data = self.data
        data = data[['Open','High','Low','Close','Volume']]
        data = data.sort_index()

        return data

    def entryDecision(self):
        data = self.dataFit()

        entry_date = data.index[6]

        return entry_date


    def lossCut(self):
        data = self.dataFit()
        entry_date = self.entryDecision()
        entry_way = self.entry_way


        if entry_way == 0:   # if long
            r = self.defR()
            cut_limit = round(float(data[data.index==entry_date].Close.values - r),2)

        if entry_way == 1:  # if short
            r = self.defR()
            cut_limit = round(float(data[data.index==entry_date].Close.values + r),2)

        return cut_limit


    def exitSt(self):
        data = self.dataFit()
        entry_date = self.entryDecision()
        entry_way = self.entry_way
        cut_limit = self.lossCut()
        r = self.defR()

        if entry_way == 0:
            for i in range(6, len(data.index)):
                if data.Low[i] < cut_limit:
                    return data.index[i]
                    break
                elif np.max(data.Close[6:i+1])-data.Close[i] > r:
                    return data.index[i]
                    break
                else:
                    exit_date = data.index[i]

            return exit_date

        else:
            for i in range(6, len(data.index)):
                if data.High[i] > cut_limit:
                    return data.index[i]
                    break
                elif data.Close[i] - np.min(data.Close[6:i + 1])  > r:
                    return data.index[i]
                    break
                else:
                    exit_date = data.index[i]

            return exit_date



    def defR(self,window=6):
        data = self.dataFit()

        ATR = []
        for i in range(1,window):
            range1 = data.High[i] - data.Low[i]
            range2 = np.abs(data.Close[i-1] - data.High[i])
            range3 = np.abs(data.Close[i-1] - data.Low[i])

            ATR = ATR + [np.max([range1,range2,range3])]

        r = round(np.mean(ATR),2)

        return r

    def result(self):
        return self.entryDecision(),self.exitSt(),self.entry_way


class calReturn:
    def __init__(self,data, decisionData):
        self.data = data
        self.decisionData = decisionData

    def profit(self):
        data = self.data
        entry_date, exit_date, decision_way = self.decisionData

        if decision_way == 0:
            entry_price = data[data.index==entry_date].Close.values
            exit_price = data[data.index==exit_date].Close.values

            pf = exit_price/entry_price-1
            return pf

        if decision_way == 1:
            entry_price = data[data.index==entry_date].Close.values
            exit_price = data[data.index==exit_date].Close.values

            pf = entry_price/exit_price-1
            return pf




if __name__ == "__main__":
    data = pd.read_csv('kospi_data.csv',index_col='Date',parse_dates=True)

    hit_ratio_list =[]
    expected_returns =[]

    for k in range(0,50):
        sample_key = np.random.randint(0, 1000, 100)
        pf_list = []

        for i in sample_key:
            a = st0901(data[i:i + 300])

            # print("entry_date : ", a.entryDecision(),",    entry_way :",a.entry_way)
            # print("exit_date  : ", a.exitSt())
            # print("entry_price:" , data[data.index==a.entryDecision()].Close.values, "exit_price :",data[data.index == a.exitSt()].Close.values,"loss_cut",a.lossCut())

            b = calReturn(data, a.result())
            pf_list = pf_list + [float(b.profit())]

            # data.Close[0:0+300].plot()
            # data.Low[0:300].plot()
            # plt.show()

        # print(pf_list)
        # print("hit_ratio : ", sum([x > 0 for x in pf_list]) / 100)
        # print(np.mean(pf_list))
        # plt.hist(pf_list, bins=10)
        # plt.show()

        hit_ratio_list = hit_ratio_list + [sum([x > 0 for x in pf_list]) / 100]
        expected_returns = expected_returns + [np.mean(pf_list)]

    print(hit_ratio_list)
    print(expected_returns)
