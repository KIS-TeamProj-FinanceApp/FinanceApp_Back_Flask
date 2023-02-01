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
df_shares = pd.read_csv('NumShares_RayDalio.csv')
df_shares.rename(columns = {'Unnamed: 0':'date'}, inplace=True)
df_shares.set_index(pd.to_datetime(df_shares['date']), inplace = True)
del df_shares['date']

stocks_use = ['MSFT', 'VZ', 'IBM', 'JNJ', 'CSCO', 'AAPL', 'ORCL', 'INTC', 'GILD', 'VTR', 'LVS', 'CAG', 'QCOM', 'EOG', 'OXY', 'RTX', 'COP', 'FCX', 'XOM', 'CVX', 'K', 'WELL', 'ED', 'MU', 'HON', 'NEM', 'APA', 'WYNN', 'PHM', 'JNPR', 'EMR', 'BA', 'MGM', 'MO', 'BKNG', 'URI', 'CCL', 'CHRW', 'TXN', 'NTAP', 'KR', 'VRSN', 'ICE', 'MAR', 'PEP', 'INTU', 'NVDA', 'UNH', 'MCK', 'MDT', 'WY', 'CI', 'MHK', 'HES', 'TMO', 'GOOGL', 'DVN', 'PXD', 'ADBE', 'BAX', 'MRO', 'DHR', 'DGX', 'DVA', 'KLAC', 'MCHP', 'LRCX', 'MET', 'NOC', 'FFIV', 'AMAT', 'PH', 'META', 'WM', 'PFE', 'LMT', 'CAH', 'IP', 'PRU', 'AIG', 'DOV', 'MSI', 'EQT', 'CTRA', 'GE', 'AFL', 'ITW', 'VLO', 'WHR', 'GWW', 'CRM', 'ACN', 'AZO', 'JBHT', 'ABC', 'ATVI', 'LLY', 'ANSS', 'EL', 'MAS', 'CB', 'ETN', 'GS', 'TRV', 'LUV', 'GLW', 'NUE', 'HD', 'CMI', 'ISRG', 'STLD', 'DUK', 'KMX', 'TEL', 'PEG', 'EBAY', 'RHI', 'PKG', 'PPG', 'LNC', 'EMN', 'FISV', 'UAL', 'SYK', 'ZION', 'NDAQ', 'LOW', 'LH', 'MTD', 'O', 'EQIX', 'SBAC', 'ROK', 'T', 'TROW', 'MMC', 'ABT', 'ALL', 'AME', 'HIG', 'LUMN', 'AKAM', 'WAB', 'GIS', 'AMZN', 'BLK', 'ADP', 'CDNS', 'BSX', 'MS', 'AEP', 'SBNY', 'PGR', 'BEN', 'UHS', 'PAYX', 'ON', 'CLX', 'BDX', 'NVR', 'SCHW', 'SLB', 'HAL', 'STT', 'IDXX', 'AMP', 'EFX', 'RL', 'BMY', 'MSCI', 'AAP', 'KO', 'SYY', 'EA', 'ORLY', 'SHW', 'TGT', 'TSN', 'PSX', 'BIIB', 'SPGI', 'MOS', 'WMT', 'SO', 'UNP', 'ETR', 'WDC', 'CCI', 'CMG', 'CTSH', 'PCAR', 'NEE', 'AXP', 'ILMN', 'GD', 'ADI', 'ELV', 'MMM', 'HST', 'ROST', 'HUM', 'FMC', 'JCI', 'COST', 'PCG', 'ADM', 'PLD', 'PSA', 'HSY', 'AMT', 'MDLZ', 'RCL', 'CF', 'SPG', 'TAP', 'PM', 'AMGN', 'LHX', 'LKQ', 'TRMB', 'CTAS', 'CNP', 'CSX', 'RSG', 'RE', 'TECH', 'IFF', 'CRL', 'FDS', 'JKHY', 'FITB', 'ALK', 'RF', 'APH', 'SNA', 'BAC', 'NWL', 'CE', 'EIX', 'LYB', 'VRSK', 'MLM', 'FLT', 'SWKS', 'AVY', 'DIS', 'SBUX', 'MCD', 'DISH', 'MRK', 'YUM', 'DRI', 'DHI', 'TXT', 'NRG', 'D', 'FDX', 'KDP', 'STZ', 'FE', 'BWA', 'RMD', 'UPS', 'WBA', 'HOLX', 'EXC', 'EW', 'DE', 'CPB', 'BXP', 'DAL', 'MNST', 'PFG', 'REGN', 'IPG', 'CVS', 'LEN', 'TJX', 'COF', 'SWK', 'F', 'SJM', 'CBRE', 'AIZ', 'HBAN', 'USB', 'SEE', 'VMC', 'TFC', 'MCO', 'FIS', 'CMA', 'L', 'APD', 'CME', 'MKC', 'JPM', 'HRL', 'IVZ', 'DFS', 'ROP', 'TPR', 'PWR', 'KMB', 'PG', 'PPL', 'CMCSA', 'CAT', 'WFC', 'XRAY', 'V', 'J', 'COO', 'C', 'RJF', 'MPC', 'WAT', 'IT', 'FTNT', 'SNPS', 'ECL', 'ARE', 'GPN', 'IRM', 'KEY', 'WTW', 'PKI', 'BALL', 'PNC', 'GL', 'FRT', 'NWSA', 'MTB', 'WRK', 'ALGN', 'HSIC', 'EXR', 'BIO', 'TSCO', 'ALLE', 'AJG', 'UDR', 'IEX', 'CPT', 'DLR', 'REG', 'KIM', 'NDSN', 'ESS', 'AEE', 'LNT', 'AWK', 'WEC', 'DTE', 'AES', 'PNW', 'ES', 'XEL', 'ATO', 'CMS', 'ALB', 'CHD', 'TSLA', 'EXPD', 'NOW', 'ROL', 'CPRT', 'CINF', 'WRB', 'TMUS', 'VFC', 'VRTX', 'EXPE', 'NTRS', 'PTC', 'DPZ', 'CL', 'APTV', 'ODFL', 'GPC', 'DG', 'DD', 'HCA', 'KHC', 'AOS', 'EQR', 'AVB', 'SRE', 'GM', 'DLTR', 'PEAK', 'POOL', 'ZBH', 'ACGL', 'CDW', 'OMC', 'HLT', 'LW', 'FAST', 'LDOS', 'ANET', 'EVRG', 'NSC', 'NI', 'ADSK', 'FTV', 'FANG', 'TT', 'AON', 'BK', 'WST', 'CARR', 'DXCM', 'STE', 'FRC', 'TFX', 'SIVB', 'MKTX', 'CFG', 'BRO', 'FOXA', 'KMI', 'INVH', 'MAA', 'BKR', 'VTRS', 'CBOE', 'BBY', 'HII', 'PAYC', 'CSGP', 'CDAY', 'PYPL', 'MA', 'LIN', 'A', 'CTVA', 'HPQ', 'KEYS', 'AMD', 'ZBRA', 'ABBV', 'AVGO', 'HPE', 'XYL', 'MRNA', 'GRMN', 'FSLR', 'IR', 'TER', 'MPWR', 'BR', 'INCY', 'QRVO', 'OKE', 'IQV', 'GEN', 'PNR', 'SEDG', 'TTWO', 'CTLT', 'TDG', 'EPAM', 'CNC', 'LYV', 'NFLX', 'TDY', 'MOH', 'TYL', 'HWM', 'ETSY', 'HAS', 'ZTS', 'NKE', 'CHTR', 'ULTA', 'OTIS', 'NXPI', 'WBD', 'DXC', 'STX', 'BBWI', 'TRGP', 'VICI', 'AAL', 'NCLH', 'PARA', 'GNRC', 'ENPH']
def get_buffet(ticker, data):
    pf = df_shares[df_shares['Sym'] == ticker]
    #pf['Amount (000)'].loc['2016-12-01'] = (pf['Amount (000)'].loc['2016-09-01'].copy() + pf['Amount (000)'].loc['2017-03-01'])/2 # 오타 수정
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
        df_fs = pd.read_csv(f'./ratio_BW/{s}_ratios.csv')
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