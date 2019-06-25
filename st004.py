#### Strategy Test04
#### 가정 : 미국장 SP500 이 상승한 다음 날에는 한국 장도 상승한다



import pandas as pd
import matplotlib.pyplot as plt

data2 = pd.read_csv("st7_data2.csv",parse_dates= True, index_col='Date')

df = data2[['kospi_o','kospi_c','sp_o','sp_c']]

df['kospi_return'] = df.kospi_c/df.kospi_o - 1
df['sp_return'] = df.sp_c/df.sp_o -1

df = df[['kospi_return','sp_return']]
df['sp_return_1Dbefore'] = df.sp_return.shift(-1)
del df['sp_return']
df.dropna(inplace=True)

df2 = df[df.sp_return_1Dbefore > 0]
totals = len(df2.index)
positive = len(df2[df2.kospi_return>0].index)
success_ratio = positive/totals
print(success_ratio)

plt.scatter(df2.sp_return_1Dbefore,df2.kospi_return)
plt.show()
