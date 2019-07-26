import pandas as pd
import datetime as dt
import time
import numpy as np
import matplotlib.pyplot as plt

class FindLowValue:
    def __init__(self,data):
        self.data = data

    def net_net_asset(self, wgt = [0.85,1,-1,-1]):
        data = self.data


        netneta_list =[]
        for i in range(0,len(data.index)):
            netneta_list = netneta_list + [float(data.cur_asset.values[i])*wgt[0]+float(data.fix_asset.values[i])*wgt[1]+float(data.cur_liab.values[i])*wgt[2]+float(data.non_cur_liab.values[i])*wgt[3]]

        data['NNAsset'] = netneta_list

        return data['NNAsset']

    def compare(self):
        market_cap = self.data['market_cap']
        netneta = self.net_net_asset()

        ratio_list = []
        for i in range(0,len(market_cap.index)):
            ratio_list  = ratio_list + [netneta[i]/float(market_cap[i])]

        df = pd.DataFrame(ratio_list,index=market_cap.index)
        df.columns = ['Ratio']
        return df



def data_cleansing(data,columns):
    dates = data['Unnamed: 0'][1:]
    dates = [dt.datetime.strptime(x,"%Y-%m-%d").date() for x in dates]

    df = pd.DataFrame(index=dates,columns={'cur_asset',
                                           'fix_asset',
                                           'cur_liab',
                                           'non_cur_liab',
                                           'market_cap'})



    df['cur_asset'][0:len(df.index)] = data[columns][1:]
    df['fix_asset'][0:len(df.index)] = data[columns+'.1'][1:]
    df['cur_liab'][0:len(df.index)] = data[columns + '.2'][1:]
    df['non_cur_liab'][0:len(df.index)] = data[columns+'.3'][1:]
    df['market_cap'][0:len(df.index)] = data[columns + '.4'][1:]

    df.index = pd.to_datetime(df.index)

    return df


def main():
    data = pd.read_csv('kr_stock_value.csv')

    cols = data.columns[1:813]

    for i in cols:
        data_selected = data_cleansing(data,i)
        data_selected.dropna(inplace=True)
        if len(data_selected.index)>=30:
            a = FindLowValue(data_selected)

            if a.compare()['Ratio'][0]>0.9:
                print(i)
                # a.compare().plot()
                # plt.title(i)
                # plt.show()

if __name__=="__main__":
    main()
