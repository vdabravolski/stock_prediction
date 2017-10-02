import pickle
import quandl
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import timedelta

# Quandl configuration
quandl.ApiConfig.api_key = 'h6tfVg1ps54v4bcpc3xz'

# Data source configuration
DATA_FOLDER = 'data/'
TIMESERIES_FOLDER = "timeseries/"
FUNDAMENTALS_FOLDER = "fundamentals/"
NEWS_FOLDER = 'news/'
NEWS_ORIG_FOLDER = '../../../seeking_alpha/output/'
SP500_FILE = DATA_FOLDER + "sp500.p"


def timeseries_data_pipeline():
    # 1. load scrapped list of SP500 companies
    with open(SP500_FILE, 'rb') as file:
        sp500_df = pickle.load(file)

    # 2. For each ticker from SP500 load it's timeseries data
    sp500_tickers = sp500_df.ticker

    # TODO: delay one hot encoding. need to decide where it's more appropriate to do.
    # #3. Prepare one-hot encoded sector id for each ticker
    # encoded_sector_id = to_categorical(sp500_df.sector_id)

    for index, ticker in enumerate(sp500_tickers):
        print("Handling ticker {0}".format(ticker))
        ticker_df = quandl.get_table('WIKI/PRICES', ticker=ticker)

        if ticker_df.shape[0] == 0:  # to handle cases when Quandle doesn't return data for a given ticker.
            print("Cannot retrieve Quandl data for {0}. Skipping...".format(ticker))
            continue

        ticker_df = _df_normalizer(ticker_df, ["open", "high", "low", "close", "volume", "ex-dividend", "split_ratio",
                       "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"])

        # 4. Augment time series data with industry sector information
        ticker_df['sector_id'] = sp500_df.sector_id.loc[index]

        # 5. Save to the disk as data frame
        ticker_df_file = DATA_FOLDER + TIMESERIES_FOLDER + "{0}_df.p".format(ticker)
        with open(ticker_df_file, "wb") as file:
            pickle.dump(ticker_df, file)

    return None


def fundamentals_data_pipeline():
    df_fund = pickle.load(open(DATA_FOLDER + "sp500_fund.p", "rb"))
    df = pickle.load(open(DATA_FOLDER + "sp500.p", "rb"))

    df_tmp = df_fund.copy()
    df_sp500 = df_tmp.merge(df, left_on='company_id',
                            right_on='company_id', how='inner')
    df_sp500 = pd.melt(df_sp500, id_vars=['company_id', 'indicator_id', 'ticker_id',
                                          'sector_id', 'sector', 'ticker'],
                       var_name='year', value_name='value')

    df_sp500 = df_sp500[pd.notnull(df_sp500['value'])]

    tmp_df = df_sp500.pivot_table(index=['company_id', 'year', 'ticker', 'sector'], columns='indicator_id',
                                  values='value')
    unpitoved_df = tmp_df.reset_index()

    indicators = ['Assets', 'LiabilitiesAndStockholdersEquity', 'StockholdersEquity',
                  'CashAndCashEquivalentsAtCarryingValue',
                  'NetCashProvidedByUsedInOperatingActivities', 'NetIncomeLoss',
                  'NetCashProvidedByUsedInFinancingActivities',
                  'CommonStockSharesAuthorized', 'CashAndCashEquivalentsPeriodIncreaseDecrease', 'CommonStockValue',
                  'CommonStockSharesIssued',
                  'RetainedEarningsAccumulatedDeficit', 'CommonStockParOrStatedValuePerShare',
                  'NetCashProvidedByUsedInInvestingActivities',
                  'PropertyPlantAndEquipmentNet', 'AssetsCurrent', 'LiabilitiesCurrent', 'CommonStockSharesOutstanding',
                  'Liabilities',
                  'OperatingIncomeLoss']
    columns_to_keep = ['company_id', 'year', 'ticker', 'sector'] + indicators

    filtered_df = unpitoved_df.loc[:, columns_to_keep]

    scaler = MinMaxScaler()
    filtered_df[indicators] = scaler.fit_transform(filtered_df.fillna(value=0)[indicators])

    ticker_df_file = DATA_FOLDER + FUNDAMENTALS_FOLDER + "norm_fund.p"
    with open(ticker_df_file, "wb") as file:
        pickle.dump(filtered_df, file)

def _df_normalizer(df, columns):
    """
    :param df: to be normalized
    :param columns: target columns
    :return: df: dataframe with normalized values in target columns
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df




def news_data_pipeline(drop_news=True, ticker_list=None):
    '''
    This method expect a dataframe with following indices: [u'content', u'datetime', u'polarity', u'subjectivity', u'ticker']
    The filename should be in format of '{ticker}_news_df.p'
    The proper dataframe is produced by developed crawler: https://github.com/vdabravolski/marketNewsCrawler

    The method:
        1. adds stock price (close date) to each news record
        2. by default drops news
        3. normalizes numeric features.

    :return:
        None. Saved the pickled dataframe (one per each ticker) to defined folder
    '''

    if ticker_list == None:  # if ticker list is Non, then we are handling all tickers which have news files
        ticker_list = []
        for root, dirs, filenames in os.walk(NEWS_ORIG_FOLDER):
            for f in filenames:
                ticker_list.append(f[0:f.find('_')])

    for ticker in ticker_list:
        try:
            news_df = pd.read_pickle(NEWS_ORIG_FOLDER + '{0}_news_df.p'.format(ticker))
        except:
            raise ("Cannot find news file for {0} in folder{1}".format(ticker, NEWS_ORIG_FOLDER))

        # 1. Get ticker price and add current close data and next date
        ticker_df = quandl.get_table('WIKI/PRICES', ticker=ticker)

        aug_news_df = news_df
        # Adding new columns with NaN values
        aug_news_df = aug_news_df.assign(close_bool=pd.Series())
        aug_news_df = aug_news_df.assign(close_bool_next_day=pd.Series())
        aug_news_df = aug_news_df.assign(close=pd.Series())


        for day in news_df.datetime.unique():
            day = pd.Timestamp(day)
            next_day = day + timedelta(1)
            previous_day = day - timedelta(1)

            close_day = ticker_df[ticker_df.date == day.date()].close.values
            close_next_day = ticker_df[ticker_df.date == next_day.date()].close.values
            close_previous_day = ticker_df[ticker_df.date == previous_day.date()].close.values


            aug_news_df.loc[aug_news_df.datetime == day, ['close_bool', 'close_bool_next_day', 'close']] \
                = _convert_labels_news(close_previous_day, close_day, close_next_day)

        # 2. Drop 'content' column
        if drop_news:
            aug_news_df = aug_news_df.drop(['content'], axis=1)

        # 3. Normalize data
        norm_news_df = _df_normalizer(aug_news_df, columns=['close'])

        with open(DATA_FOLDER+NEWS_FOLDER + "{0}_df.p".format(ticker), "wb") as file:
            pickle.dump(norm_news_df, file)



def _convert_labels_news(close_previous_day, close_day, close_next_day):
    # handling cases when we don't have prices for a given day
    close_day = 0 if len(close_day) == 0 else close_day[0]
    close_next_day = 0 if len(close_next_day) == 0 else close_next_day[0]
    close_previous_day = 0 if len(close_previous_day) == 0 else close_previous_day[0]

    diff_bool = 1 if close_previous_day < close_day else 0
    diff_bool_next_day = 1 if close_day < close_next_day else 0

    return diff_bool, diff_bool_next_day, close_day

def convert_data_to_batch_timesteps(data, batch_size, timesteps, features, time_range=None):
    """
    This method takes as an input sequence of 2D timeseries data of shape (nb_samples, features)
    and converst it to 3D data pof shape (nb_samples, timesteps, features), where timesteps define a slice of timeseries data.
    """

    if time_range is not None:  # check if a specific time range needed. Otherwise, process all data.
        data_time_slice = data.loc[(data['date'] > time_range[0]) & (data['date'] < time_range[1])]
    else:
        data_time_slice = data

    # Dropping "date" column as we no longer need it.
    data_time_slice = data_time_slice.drop(labels='date', axis=1)

    augmented_df = pd.DataFrame()  # initate new
    # for idx in range(data_time_slice.shape[0] - timesteps - 1):
    for idx in range(data_time_slice.shape[0] - timesteps):
        data_sample = data_time_slice[idx: idx + timesteps]
        augmented_df = augmented_df.append(data_sample)
    augmented_df = augmented_df.reset_index(drop=True)

    sample_size = augmented_df.shape[0]
    trimmed_idx = math.floor(sample_size / (batch_size * timesteps)) * (batch_size * timesteps)
    resized_data = np.reshape(augmented_df[:trimmed_idx].values, (-1, timesteps, features))

    return resized_data # TODO: return also from here a date range, so we can build a new data upon.


def resize_data_for_batches(data, batch_size):
    """this method is used to resize input data to fit selected batch size"""
    sample_size = data.shape[0]
    new_sample_size = math.floor(sample_size / batch_size) * batch_size

    return data[:new_sample_size]


def convert_ts_to_categorial(data_df, timesteps):
    """
    :param
        data_df: - input data frame which contains timeseries float variable ('close' column) and datetime variable ('date' column)
        timesteps - parameters which defines shift in time. Refer to LSTM implementations for details.
    :return:
    target_df - dataframe with following columns
        'diff_bool' -  boolean difference in value between t+1 and t times.
        'value' - contains value in t+1 time.
        'date' - contains date
    """

    diff_bool = data_df.close[(timesteps + 1):].reset_index(drop=True) > data_df.close[timesteps:-1].reset_index(
        drop=True)
    diff_bool = diff_bool.astype(int)  # converting boolean value to integer
    value = data_df.close[(timesteps + 1):].reset_index(drop=True)
    date = data_df.date[(timesteps + 1):].reset_index(drop=True)
    target_df = pd.concat([diff_bool.rename('close_bool'), value, date], axis=1).reset_index(drop=True)

    return target_df


if __name__ == '__main__':
    news_data_pipeline(ticker_list=['AAPL'])
