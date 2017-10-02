import requests
import re
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from datetime import datetime

'''
This set of methods allows to retrieve list of SP500 companies, basic information about them
and store the outcomes as pickled DataFrame for further processing.
'''


SITE = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START = datetime(2017, 5, 20)
END = datetime.today().utcnow()
SP_file = 'sp500_3.p'
SP_fund_file = 'sp500_fund_3.p'

#TOKENS and APIs
US_FUND_TOKEN = "YlU6wgW1_ZNhf0LZb7DlSA"
print("reading the list of available indicators...")
META_API = "https://api.usfundamentals.com/v1/indicators/xbrl/meta?token={0}".format(US_FUND_TOKEN)

COMPANIES_API = "https://api.usfundamentals.com/v1/companies/xbrl?" \
                "format=json&token={0}&companies=".format(US_FUND_TOKEN)
FUNDAMENTALS = ['Assets', 'LiabilitiesAndStockholdersEquity', 'StockholdersEquity', 'CashAndCashEquivalentsAtCarryingValue',
                'NetCashProvidedByUsedInOperatingActivities', 'NetIncomeLoss', 'NetCashProvidedByUsedInFinancingActivities',
                'CommonStockSharesAuthorized', 'CashAndCashEquivalentsPeriodIncreaseDecrease', 'CommonStockValue', 'CommonStockSharesIssued',
                'RetainedEarningsAccumulatedDeficit', 'CommonStockParOrStatedValuePerShare', 'NetCashProvidedByUsedInInvestingActivities',
                'PropertyPlantAndEquipmentNet', 'AssetsCurrent', 'LiabilitiesCurrent', 'CommonStockSharesOutstanding', 'Liabilities',
                'OperatingIncomeLoss']
INDICATORS_API = "https://api.usfundamentals.com/v1/indicators/xbrl?indicators={0}" \
                 "&token={1}&companies=".format(','.join(FUNDAMENTALS), US_FUND_TOKEN)
print("done with indicators...")


def scrape_SP_list(site):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(site, headers=hdr)
    soup = BeautifulSoup(req.content, "html.parser")

    table = soup.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict({'ticker':[],"sector":[]})
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            sector = str(col[3].string.strip()).lower().replace(' ', '_')
            ticker = str(col[0].string.strip())
            sector_tickers['ticker'].append(ticker)
            sector_tickers['sector'].append(sector)

    ticker_df = pd.DataFrame(sector_tickers)
    ticker_df['ticker'] = ticker_df['ticker'].astype('category')
    ticker_df['ticker_id'] = ticker_df['ticker'].cat.codes

    ticker_df['sector'] = ticker_df['sector'].astype('category')
    ticker_df['sector_id'] = ticker_df['sector'].cat.codes

    pickle.dump(ticker_df, open(SP_file, 'wb'))
    return ticker_df

def _get_CIK_by_ticker(ticker):
    URL = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'.format(ticker)
    CIK_RE = re.compile(r'.*(\d{10}).*')

    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(URL, headers=hdr)
    soup = BeautifulSoup(req.content, "html.parser")
    span = soup.find('span',{'class': 'companyName'})

    try: # if ticker doesn't have CIK on edgar
        results = CIK_RE.findall(span.text)
    except:
        results = ['']

    if len(results):
        return str(results[0])
    return str(results)

def _get_fund_by_CIK(cik):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = requests.get(INDICATORS_API+cik, headers=hdr)
    print(req.text)
    return req.text


def try_load_SP():
    try:
        with open(SP_file, "rb") as f:
            ticker_df = pickle.load(f)
    except (OSError, IOError) as e:
        ticker_df = scrape_SP_list(SITE)
    return ticker_df


def augment_w_CIK(ticker_df):
    # now let's check if ticker already have CIK columns
    new_ticker_df = ticker_df.copy()
    for index, row in ticker_df.iterrows():
        cik = _get_CIK_by_ticker(row['ticker'])
        new_ticker_df.loc[index, 'CIK'] = cik
    with open(SP_file,'wb') as f:
        pickle.dump(new_ticker_df, f)

    return ticker_df


def create_fund_df():
    ticker_df = try_load_SP() # try load pre-existing file with ticker

    fund_df = pd.concat((pd.read_csv(INDICATORS_API+row['CIK']) for index, row in ticker_df.iterrows()))

    with open(SP_fund_file, 'wb') as f:
        pickle.dump(fund_df, f)
        print(fund_df.reset_index())
    return fund_df


def get_snp500():
    sector_tickers = scrape_SP_list(SITE)
    augment_w_CIK(sector_tickers)
    create_fund_df()


if __name__ == '__main__':
    get_snp500()