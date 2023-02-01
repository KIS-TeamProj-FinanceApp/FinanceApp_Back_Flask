import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import requests
import warnings
warnings.filterwarnings('ignore')

df_macro = pd.read_csv('macro_data_interpolated.csv')
df_macro.rename(columns = {'Unnamed: 0':'date'}, inplace=True)
df_macro.set_index(pd.to_datetime(df_macro['date']), inplace = True)
del df_macro['date']
df_shares = pd.read_csv('NumShares_Buffet.csv')
df_shares.rename(columns = {'Unnamed: 0':'date'}, inplace=True)
df_shares.set_index(pd.to_datetime(df_shares['date']), inplace = True)
del df_shares['date']

stocks_use = ['WFC', 'KO', 'AXP', 'IBM', 'WMT', 'PG', 'XOM', 'USB', 'DVA', 'GS', 'MCO', 'GHC', 'GM', 'BK', 'COP', 'PSX', 'NOV', 'MTB', 'VRSN', 'VZ', 'COST', 'DE', 'V', 'GL', 'LBTYA', 'MA', 'LBTYK', 'GE', 'SNY', 'VRSK', 'JNJ', 'MDLZ', 'UPS', 'LEE', 'LBRDA', 'QSR', 'FWONK', 'DNOW', 'KHC', 'AAPL', 'AXTA', 'KMI', 'LILA', 'LILAK', 'CHTR', 'LUV', 'DAL', 'AAL', 'UAL', 'SIRI', 'BAC', 'SYF', 'TEVA', 'STOR', 'JPM', 'PNC', 'AMZN', 'TRV', 'KR', 'OXY', 'BIIB', 'RH', 'CVX', 'ABBV', 'BMY', 'MRK', 'AON', 'TMUS', 'MMC', 'ATVI', 'HPQ', 'C', 'PARA', 'CE', 'MCK', 'MKL', 'ALLY', 'FND', 'FWONA', 'T', 'ORCL', 'PFE', 'TSM', 'LPX', 'JEF']
def get_buffet(ticker, data):
    pf = df_shares[df_shares['Sym'] == ticker]
    pf['Amount (000)'].loc['2016-12-01'] = (pf['Amount (000)'].loc['2016-09-01'].copy() + pf['Amount (000)'].loc['2017-03-01'])/2 # 오타 수정
    pf['Diff'] = pf['Amount (000)'] - pf['Amount (000)'].shift(1) # 이전꺼와의 차이
    return pf['Diff']

sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500 = pd.read_html(sp_url, header=0)[0]
notsp = []
stocks_saved = []
for i in stocks_use:
    if i in sp500['Symbol'].values:
        stocks_saved.append(i)
    else:
        notsp.append(i)
sp500 = sp500.set_index('Symbol')

sec_dict = {'NOV':'Materials','MKL':'Materials','GHC':'Communication Services','SIRI':'Communication Services',
            'LBTYA':'Real Estate','LBTYK':'Real Estate','RH':'Consumer Discretionary','FWONA':'Utilities',
            'TSM':'Information Technology','LPX':'Industrials'}

url = 'https://www.sec.gov/files/company_tickers_exchange.json'
headers = {'User-Agent': 'Mozilla'}
res = requests.get(url, headers=headers)
cik_list = res.json()
company_info = cik_list['data']
tickers = dict()
for p in company_info:
    if p[2] in stocks_use:
        tickers[p[2]] = [p[1],p[3]] # 이름, 거래소

def OLS_regression():
    regression_results = []
    for s in stocks_use:
        df_fs = pd.read_csv(f'./ratio_db/{s}_ratios.csv')
        df_fs.set_index(pd.to_datetime(df_fs['date']), inplace = True)
        del df_fs['date']

        y_temp = get_buffet(s, df_shares)
        X = df_macro.join(df_fs).interpolate()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.interpolate()
        #scaler = StandardScaler()
        #scaler.fit_transform(X)
        data = X.join(y_temp).fillna(method = 'ffill')
        Y = data.iloc[:,-1]
        Y = Y.replace([np.inf, -np.inf], np.nan)
        Y = Y.interpolate()
        F_Y = Y.shift(-1).dropna() # 미래를 대상으로 fitting
        #F_Y_use = F_Y.iloc[:-1]
        X_use = X.loc['2014-02-01':'2022-11-01']
        X_test = X.loc['2022-12-01':]

        try:
            reg = LinearRegression().fit(X_use, F_Y)
            y_pred = reg.predict(X_test)
            if s in stocks_saved:
                regression_results.append([s, int(y_pred[0]), sp500.loc[s]['Security'], tickers[s][1], sp500.loc[s]['GICS Sector']]) # ticker, 예상치, 이름, 거래소, 산업분류
            else:
                regression_results.append([s, int(y_pred[0]), tickers[s][0], tickers[s][1], sec_dict[s]])
        except:
            pass
    regression_results.sort(key = lambda x:x[1], reverse=True)
    
    return regression_results # [[Ticker, 매매수량, 회사명, 거래소, GICS Sector], .... [Ticker, 매매수량, 회사명, 거래소, GICS Sector]]
print(OLS_regression())