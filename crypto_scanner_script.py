# %%
# Import the required libraries.
import psycopg2
from psycopg2 import pool
import pandas as pd
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import requests
import squarify
import logging
import gc
import plotly.express as px
import json
import threading
import io
import ta

# %%
# Variables for connection.
pd.options.mode.copy_on_write = True

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

db_config_Test = {
    'host': '000.000.00.000',
    'dbname': 'crypto_market',
    'user': 'user',
    'password': 'password',
    'port': 5432
}

sql_query_live = "SELECT * FROM data_30m;"  # Live data

cwd = os.getcwd()

# %%
# Decorator for counting time of a process.


def count_seconds(func):
    """
    Decorator that counts the time taken for a function to execute.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with time counting.
    """
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(
            f"Time taken for {func.__name__}: {end - start:.2f} seconds")
        return result
    return inner

# %%
# Function to interact with the symbols APIs database counting time of execution.


@count_seconds
def psycopg2_connect(db_config):
    """
    Create a connection from PostgreSQL using psycopg2.

    Args:
        db_config (dict): Dictionary containing PostgreSQL connection details.
    """
    try:
        # Establish the database connection
        connection = psycopg2.connect(
            host=db_config['host'],
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            port=db_config.get('port', 5432)
        )
        connection.autocommit = True
        return connection
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None


@count_seconds
def fetch_table_with_psycopg2(connection, sql_query, batch_size=1000):
    """
    Fetch data from PostgreSQL using psycopg2 and load it into a pandas DataFrame.

    Args:        
        connection (psycopg2.extensions.connection): The connection object to the PostgreSQL database.
        sql_query (str): The SQL query to execute.
        batch_size (int, optional): The number of rows to fetch per batch. Default is 1000.

    Returns:
        pd.DataFrame: The fetched data as a pandas DataFrame.
    """
    try:
        # Fetch data using a cursor
        logger.info(f"Starting query: {sql_query}")
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            columns = [desc[0]
                       for desc in cursor.description]  # Get column names
            data = []
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                data.extend(batch)

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


# %%
# Functions for preprocessing dataframes.
def reformat_dates(df):
    """
    Changes the raw datetimes to pd.datetime in all temporal columns.

    Args:
        df (DataFrame): the dataframe to be processed.

    Returns:
        DataFrame: the same DataFrame processed.
    """
    # Convert 'event_time' and 'datetime' to datetime format.
    df['event_time'] = pd.to_datetime(df['event_time'], unit='ms', utc=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)

    return df


def tokens_to_uppercase(df):
    """
    Reformat the column tokens to uppercase.

    Args:
        df (DataFrame): the dataframe to process.

    Returns:
        Dataframe: the dataframe with tokens in uppercase.
    """
    if not df.empty:
        df['token'] = df['token'].str.upper()

    return df


def sorting_by_columns(df, columns):
    """
    Sort the dataframes by the given columns.

    Args:
        df (Dataframe): the dataframe to process.
        columns (list): the list of columns to sort the dataframe.

    Returns:
        Dataframe: the dataframe sorted by the columns.
    """
    df = df.sort_values(by=columns, ignore_index=True)

    return df
# %%
# Function to remove duplicates from dataframes.


def remove_duplicates_from_df(df):
    """
    Remove duplicates in the columns 'datetime' and 'token' for a Dataframe, and keep
    the last data found.

    Args:
        df (DataFrame): dataframe to process.

    Returns:
        DataFrame: resulting dataframe.
    """
    # Drop duplicates in 'datetime' and 'token' keeping last occurrence.
    df_fixed = df.drop_duplicates(
        subset=['datetime', 'token'], keep='last').reset_index(drop=True)

    return df_fixed

# %%
# Functions to calculate WAD.


def calculate_wad(df, last_strength):
    """ 
    Calculate the Williams Accumulation/Distribution (WAD).

    This function calculates the WAD for a given DataFrame containing historical market data.
    The WAD is a cumulative total of the relationship between the close price and the true high/low prices.

    Args:
        df (pd.DataFrame): DataFrame sorted historically with columns 'high', 'low', and 'close'.
        last_strength (float): The last WAD value to continue the calculation.

    Returns:
        pd.Series: A Series with the WAD calculation for each row.
    """
    true_high = np.maximum(df['high'], df['close'].shift(1))
    true_low = np.minimum(df['low'], df['close'].shift(1))
    mom = df['close'].diff()

    gain = np.where(mom > 0, df['close'] - true_low,
                    np.where(mom < 0, df['close'] - true_high, 0))

    # Insert the last WAD to continue the calculation.
    if len(gain) > 0 and len(last_strength) > 0:
        gain[0] = float(last_strength.iloc[0])

    wad = np.cumsum(gain)

    return wad


def calculate_sma_wad(df, period, df_strength):
    """
    Calculate the simple moving average of the WAD.

    This function calculates the simple moving average (SMA) of the WAD for a given period.
    It merges the WAD data with the strength data and calculates the rolling mean.

    Args:
        df (pd.DataFrame): DataFrame sorted historically with a 'wad' column.
        period (int): Number of periods to calculate the mean.
        df_strength (pd.DataFrame): DataFrame with strength data for the token.

    Returns:
        np.ndarray: An array with the SMA values for the WAD.

    """
    # Adjust the latest data storage in the tables and the data from the closed cancles to sync.
    df_sma = pd.merge(df_strength[['datetime', 'strength']], df[[
                      'datetime', 'wad']], on='datetime', how='outer')
    df_sma['wad'] = df_sma['wad'].fillna(df_sma['strength'])
    df_sma.drop('strength', axis=1, inplace=True)

    df_sma['sma'] = df_sma['wad'].rolling(window=period).mean()

    return df_sma.iloc[-(len(df)):]['sma'].values


def analyze_wad_for_each_token(df, df_strength):
    """
    Calculate the Williams Accumulation/Distribution (WAD) and related indicators for each token.

    This function takes a DataFrame with market data and a DataFrame with strength data, and calculates
    the WAD, SMAs for WAD (57 and 25 periods), and various indicators such as WAD crossing above/below
    SMAs and whether WAD is above/below SMAs.

    Args:
        df (pd.DataFrame): A DataFrame with all market data sorted historically.
        df_strength (pd.DataFrame): A DataFrame with strength data for the token.

    Returns:
        pd.DataFrame: A DataFrame with all tokens' WAD data and related indicators sorted historically.
    """
    last_strength = df_strength[df_strength['datetime'] == df.iloc[0]
                                ['datetime']]['strength'] if len(df_strength) > 0 else pd.Series(0)

    # Calculate WAD
    df['wad'] = calculate_wad(df, last_strength)

    # Calculate SMAs for WAD (57 and 25 periods)
    df['sma57'] = calculate_sma_wad(
        df, 57, df_strength) if len(df_strength) == 58 else 0.0
    df['sma25'] = calculate_sma_wad(
        df, 25, df_strength) if len(df_strength) == 58 else 0.0

    # Adjust the data type to avoid errors.
    df['sma57'] = df['sma57'].astype('float64')
    df['sma25'] = df['sma25'].astype('float64')

    # Detect WAD crossing above/below SMA57
    df['wad_cross_up57'] = (df['wad'] > df['sma57']) & (
        df['wad'].shift(1) < df['sma57'].shift(1))
    df['wad_cross_down57'] = (df['wad'] < df['sma57']) & (
        df['wad'].shift(1) > df['sma57'].shift(1))

    # Detect WAD crossing above/below SMA25
    df['wad_cross_up25'] = (df['wad'] > df['sma25']) & (
        df['wad'].shift(1) < df['sma25'].shift(1))
    df['wad_cross_down25'] = (df['wad'] < df['sma25']) & (
        df['wad'].shift(1) > df['sma25'].shift(1))

    # Detect whether WAD is above/below SMA57 and SMA25
    df['wad_above_57'] = df['wad'] > df['sma57']
    df['wad_below_57'] = df['wad'] < df['sma57']
    df['wad_above_25'] = df['wad'] > df['sma25']
    df['wad_below_25'] = df['wad'] < df['sma25']

    # Delete the row added.
    df = df.drop(0).reset_index(drop=True)

    return df


# Define clasification orden for sorting.
class_order = {
    'Just Crossed Above': 1,
    'Above EMA': 2,
    'Below EMA': 3,
    'Just Crossed Below': 4
}

# %%
# Function to resample the 30-min data to a larger timeframe


def resample_candle_data(df, timeframe):
    resampled_df = df.set_index('datetime').groupby('token').resample(timeframe).agg({
        'event_time': 'first',  # First event time in the window
        'open': 'first',  # First open price in the window
        'high': 'max',    # Maximum high price in the window
        'low': 'min',     # Minimum low price in the window
        'close': 'last',  # Last close price in the window
        'volume': 'sum',   # Sum of the volumes in the window
        'iscandleclosed':  'last'
    }).reset_index()

    return resampled_df

# %%
# Function to generate a complete dataframe from WAD data for above and below signals at 25 and 57 periods.


def generate_signal_df(df, signal):
    """
    Create a main dataframe with  all the results of the WAD calculations.

    Args:
        df (DataFrame): the df_wad with all the boolean data.
        signal (string): the time lapse between rows or the deltatime between datetime data.

    Returns:
        Dataframe: the resulting DataFrame for all trades calculation.
    """
    df_signal = pd.DataFrame()

    # Generate a complete datetime index (with 30-minute intervals)
    df_signal['datetime'] = pd.date_range(
        start=df['datetime'].min(), end=df['datetime'].max(), freq=signal)

    condition_column_list = ['wad_below_25', 'wad_above_25', 'wad_cross_down25', 'wad_cross_up25', 'wad_below_57', 'wad_above_57',
                             'wad_cross_down57', 'wad_cross_up57']

    for condition_column in condition_column_list:
        # Filter the data based on the condition.
        df_filtered = df[df[condition_column] == True]

        # Group by datetime and concatenate tokens for each datetime
        grouped = df_filtered.groupby('datetime')['token'].apply(
            lambda tokens: ', '.join(tokens)).reset_index()
        grouped.columns = ['datetime', condition_column]

        # Merge the grouped tokens with the complete datetime range to fill missing dates with 'None'
        df_signal = pd.merge(df_signal, grouped, on='datetime', how='left')

    # Rename columns to match the desired output format
    df_signal.columns = ['DateTime', 'Below_25', 'Above_25', 'Cross_Down_25',
                         'Cross_Up_25', 'Below_57', 'Above_57', 'Cross_Down_57', 'Cross_Up_57']

    return df_signal

# %%
# Functions to operate token group entries.


def array_to_string(array):
    """Concatenates in a single string all tokens presented in a list and separete them with a ', '.

    Args:
        array (list): list with tokens.

    Returns:
        string: a string with tokens.
    """
    result_str = ", ".join(array)
    return result_str


def string_to_array(tokens):
    """Splits a string made of tokens by expresion ', ' into a list.

    Args:
        tokens (string): string of tokens separated by ', '.

    Returns:
        list: list with all tokens.
    """
    tokens_array = tokens.split(", ") if type(tokens) == str else []
    return tokens_array


def subtract_token_entries(group1, group2):
    """Subtract tokens present in 'group2' from 'group1' and return a list with the result.

    Args:
        group1 (list, required): group of tokens.
        group2 (list, required): group of tokens.

    Returns:
        list: lists of the tokens as individual strings.
    """
    result = []

    for token in group1:
        if not token in group2:
            result.append(token)

    return result


def intersection_token_entries(group1, group2):
    """Find the common tokens in 'group1' and 'group2' and return a list with the result.

    Args:
        group1 (list, required): group of tokens.
        group2 (list, required): group of tokens.

    Returns:
        list: lists of the tokens as individual strings.
    """
    result = []

    for token in group1:
        ok = token in group2
        if ok:
            result.append(token)

    return result

# %%
# Functions to find trade flags.


def strong_buy_tokens(row_data_signal):
    """Find the strong buy tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.

    Returns:
        list: all the tokens with a strong buy flag.
    """
    return intersection_token_entries(
        string_to_array(row_data_signal['Cross_Up_25']),
        string_to_array(row_data_signal['Cross_Up_57'])
    )


def medium_buy_tokens(row_data_signal, strong_result):
    """Find the medium buy tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.
        strong_result (list): the resulting list for strong buy flag.

    Returns:
        list: all the tokens with a medium buy flag.
    """
    return subtract_token_entries(
        intersection_token_entries(
            string_to_array(row_data_signal['Above_25']),
            string_to_array(row_data_signal['Cross_Up_57'])
        ),
        strong_result
    )


def first_call_buy_tokens(row_data_signal):
    """Find the first call buy tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.

    Returns:
        list: all the tokens with a first call buy flag.
    """
    return intersection_token_entries(
        string_to_array(row_data_signal['Above_25']),
        string_to_array(row_data_signal['Below_57'])
    )


def old_buy_tokens(row_data_signal, strong_result, medium_result):
    """Find the old buy tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.
        strong_result (list): the resulting list for strong buy flag.
        medium_result (list): the resulting list for medium buy flag.

    Returns:
        list: all the tokens with an old buy flag.
    """
    return subtract_token_entries(
        intersection_token_entries(
            string_to_array(row_data_signal['Above_25']),
            string_to_array(row_data_signal['Above_57'])
        ),
        medium_result + strong_result
    )


def strong_sell_tokens(row_data_signal):
    """Find the strong sell tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.

    Returns:
        list: all the tokens with a strong sell flag.
    """
    return intersection_token_entries(
        string_to_array(row_data_signal['Cross_Down_25']),
        string_to_array(row_data_signal['Cross_Down_57'])
    )


def medium_sell_tokens(row_data_signal, strong_result):
    """Find the medium sell tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.
        strong_result (list): the resulting list for strong buy flag.

    Returns:
        list: all the tokens with a medium sell flag.
    """
    return subtract_token_entries(
        intersection_token_entries(
            string_to_array(row_data_signal['Below_25']),
            string_to_array(row_data_signal['Cross_Down_57'])
        ),
        strong_result
    )


def first_call_sell_tokens(row_data_signal):
    """Find the first call sell tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.

    Returns:
        list: all the tokens with a first call sell flag.
    """
    return intersection_token_entries(
        string_to_array(row_data_signal['Below_25']),
        string_to_array(row_data_signal['Above_57'])
    )


def old_sell_tokens(row_data_signal, strong_result, medium_result):
    """Find the old sell tokens in a row of data from a signal df.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formula.
        strong_result (list): the resulting list for strong buy flag.
        medium_result (list): the resulting list for medium buy flag.

    Returns:
        list: all the tokens with an old sell flag.
    """
    return subtract_token_entries(
        intersection_token_entries(
            string_to_array(row_data_signal['Below_25']),
            string_to_array(row_data_signal['Below_57'])
        ),
        medium_result + strong_result
    )


def find_trades_flags(row_data_signal):
    """Find all long and short trades flags for a row and create two lists each for trade type.

    Args:
        row_data_signal (DataFrame Series): the row to feed the formulas and plots.        

    Returns:
        tuple: two lists of lists, first for long data and the second for short data.
    """
    # * FIND LONG TRADES FLAGS.
    strong_buy = sorted(strong_buy_tokens(row_data_signal))  # Strong Buy
    medium_buy = sorted(medium_buy_tokens(
        row_data_signal, strong_buy))  # Medium Buy
    first_call_buy = sorted(first_call_buy_tokens(
        row_data_signal))  # First Call Buy
    old_buy = sorted(old_buy_tokens(
        row_data_signal, strong_buy, medium_buy))  # Old Buy

    # Build the Long Trades treemap data.
    treemap_long_data = [old_buy, first_call_buy, medium_buy, strong_buy]

    # * FIND SHORT TRADES FLAGS.
    strong_sell = sorted(strong_sell_tokens(row_data_signal))  # Strong Sell
    medium_sell = sorted(medium_sell_tokens(
        row_data_signal, strong_sell))  # Medium Sell
    first_call_sell = sorted(first_call_sell_tokens(
        row_data_signal))  # First Call Sell
    old_sell = sorted(old_sell_tokens(
        row_data_signal, strong_sell, medium_sell))  # Old Sell

    # Build the Short Trades treemap data.
    treemap_short_list = [old_sell, first_call_sell, medium_sell, strong_sell]

    return (treemap_long_data, treemap_short_list)

# %%
# Function to plot trade opportunities in a treemap.


@count_seconds
def build_plot(data_array, data_type, row_type, signal_type, color_list, color_list_2):
    """
    Build a tree map plot for 4 categories.

    Args:
        data_array (list): list of four lists,each one for a trade flag; strong, medium, first call and old.
        data_type (string): 'long'|'short' the type of trade of the data.
        row_type (string): 'live'|'one_previous'|'two_previuos'|'three_previous' the type of row is being passed to the formula,
        is used to name the plots.
        signal_type (string): the signal code '30m'|'4h'|'1d'.
        color_list (list): long_color_list | short_color_list. The primary colors for the container boxes.
        color_list_2 (list): long_color_list | short_color_list. The secundary colors for the token boxes.
    """
    total_tokens = len(
        data_array[0] + data_array[1] + data_array[2] + data_array[3])

    _, ax = plt.subplots(figsize=(6, 6))

    # Treemap Section Containers
    sectors_number = 4
    ticks_pos = []
    size_list = [len(data_array[0])/total_tokens, len(data_array[1])/total_tokens, len(data_array[2]) /
                 total_tokens, len(data_array[3])/total_tokens] if total_tokens != 0 else [0.25, 0.25, 0.25, 0.25]
    size_factor = 0.5
    line_width = 0.02
    sector_delta_y_pos = (
        0.48 - (((line_width/2) * sectors_number))) / sectors_number

    for i in range(sectors_number):
        x = 0.01
        y = (sector_delta_y_pos * i) + line_width/2 + \
            ((line_width/2) * i) + size_factor*sum(size_list[:i])
        dx = 0.98
        dy = sector_delta_y_pos + size_factor*size_list[i]
        ticks_pos.append(y + dy/2)

        ax.add_patch(FancyBboxPatch((x, y),
                                    dx,
                                    dy,
                                    boxstyle="round,pad=-0.0040,rounding_size=0.015",
                                    alpha=0.5,
                                    linewidth=line_width*100,
                                    color=color_list[i]))

        if data_array[i] != []:
            # Treemap Inner Section.
            values = [1 for _ in data_array[i]]

            values = squarify.normalize_sizes(values, dx - 0.02, dy - 0.02)

            inner_rects = squarify.squarify(
                values, x + 0.01, y + 0.01, dx - 0.02, dy - 0.02)

            for j, rect in enumerate(inner_rects):
                x = rect['x']
                y = rect['y']
                dx = rect['dx']
                dy = rect['dy']
                fontsize = 30 * dx

                if dx > 0.02:
                    x += dx/10
                    dx -= dx/5
                if dy > 0.02:
                    y += dy/10
                    dy -= dy/5

                if dx < 0.02:
                    if len(data_array[i][j]) < 4:
                        fontsize = 90 * dx
                    else:
                        fontsize = 70 * dx
                elif dx < 0.25:
                    if len(data_array[i][j]) < 4:
                        fontsize = 80 * dx
                    else:
                        fontsize = 60 * dx
                elif dx < 0.5:
                    if len(data_array[i][j]) < 7:
                        fontsize = 45 * dx
                    else:
                        fontsize = 25 * dx
                else:
                    if len(data_array[i][j]) < 7:
                        fontsize = 25 * dx
                    else:
                        fontsize = 10 * dx

                ax.add_patch(FancyBboxPatch((x, y),
                                            dx,
                                            dy,
                                            alpha=0.5,
                                            boxstyle=f"round,pad=-0.004,rounding_size={0.025/len(data_array[i])}",
                                            color=color_list_2[i]))
                ax.text(x + dx / 2, y + dy / 2, data_array[i][j] if '1000' not in data_array[i][j] else data_array[i]
                        [j][:4]+'\n'+data_array[i][j][4:], va='center', ha="center", fontsize=fontsize, fontname='monospace')

    # Customize ploting style
    if data_type == 'long':
        plt.yticks(ticks_pos, labels=['Old\nBuy', 'First Call\nBuy', 'Medium\nBuy', 'Strong\nBuy'],
                   rotation=90, va='center', ha='center', linespacing=0.9, fontweight='bold', fontsize=7)
        ax.tick_params(axis='y', colors='#45b39d', pad=15)
        ax.spines[['left']].set_color('#45b39d')

        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(color_list[i])

        plt.title('LONG TRADE FLAGS', fontweight='bold',
                  c='#117a65', fontsize=20)
    else:
        plt.yticks(ticks_pos, labels=['Old\nSell', 'First Call\nSell', 'Medium\nSell', 'Strong\nSell'],
                   rotation=90, va='center', ha='center', linespacing=0.9, fontweight='bold', fontsize=7)
        ax.tick_params(axis='y', colors='#922b21', pad=15)
        ax.spines[['left']].set_color('#922b21')

        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(color_list[i])

        plt.title('SHORT TRADE FLAGS', fontweight='bold',
                  c='#922b21', fontsize=20)

    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', colors='white')
    plt.xticks(visible=False)

    # Save image.
    plt.savefig(
        fr'{cwd}/plots/{signal_type}_{row_type}_{data_type}_trade_treemap.png', dpi=600)
    plt.close()


def update_date(current_date):
    """
    Post a date string in the flask server to communicate with the frontend.

    Args:
        current_date (string): live date in format string to update.
    """
    response = requests.post(
        'http://000.000.00.000:5000/update-date', json={"date": current_date})
    if response.status_code == 200:
        print("Date updated successfully")
    else:
        print("Failed to update date")


def update_status(current_status):
    """
    Post a status string in the flask server to communicate with the frontend.

    Args:
        current_status (string): status string to update.
    """
    response = requests.post(
        'http://000.000.00.000:5000/update-status', json={"status": current_status})
    if response.status_code == 200:
        print("Status updated successfully")
    else:
        print("Failed to update status")


def update_titles(current_titles):
    """
    Post a list string in the flask server to communicate with the frontend.

    Args:
        current_titles (dict): titles string to update.
    """
    sig = list(current_titles.keys())[0]
    for i, title in enumerate(current_titles[sig]):
        current_titles[sig][i] = title.strftime("%Y-%m-%d %H:%M:%S")

    response = requests.post(
        'http://000.000.00.000:5000/update-titles', json={"titles": current_titles})
    if response.status_code == 200:
        print("Status updated successfully")
    else:
        print("Failed to update status")


# %%
# Define color variables for plotting.
# Long Color Variables
long_color_list = ['#abebc6', '#58d68d', '#28b463', '#1d8348']
long_color_list_2 = ['#73c6b6', '#16a085', '#117a65', '#0b5345']

# Short Color Variables
short_color_list = ['#f5b7b1', '#ec7063', '#cb4335', '#943126']
short_color_list_2 = ['#d98880', '#c0392b', '#922b21', '#641e16']


#! START THE PROCESS
# %%
# Start a connection with MVP and a pool connection with Test.
conn_pool = pool.ThreadedConnectionPool(1, 10, **db_config_Test)
connection_Test = conn_pool.getconn()

update_status('Service On: Getting new data')

# %%
# Verify the live data is complete.
len_tokens_A = 0
len_tokens_B = 1

while len_tokens_A != len_tokens_B:
    df_live_data_A = fetch_table_with_psycopg2(connection_Test, sql_query_live)
    len_tokens_A = df_live_data_A['token'].nunique()

    time.sleep(10)

    df_live_data_B = fetch_table_with_psycopg2(connection_Test, sql_query_live)
    len_tokens_B = df_live_data_B['token'].nunique()

    logger.info(f'A: {str(len_tokens_A)}. B: {str(len_tokens_B)}')

    if len_tokens_A != len_tokens_B:
        time.sleep(10)

# Preprocess live data.
df_live_data = reformat_dates(df_live_data_B)
df_live_data = sorting_by_columns(df_live_data, 'event_time')

# Find unique tokens in live data.
tokens = df_live_data['token'].unique()
len_tokens = len(tokens)
# %%
# Find unique tokens in live data and divide them in two groups.
tokens_half = len_tokens//2
tokens_A = tokens[:tokens_half]
tokens_B = tokens[tokens_half:]

# %%
# Get strength and create new Strength dataframes.

# Request the most recent date in strength tables.


def get_timestamps(signal):
    """
    Gets the maximum timestamp value for a given signal.

    Args:
        signal (str): The name of the signal for which timestamp are being retrieved.

    Returns:
        int : the maximum date found in the table in EPOCH miliseconds format
    """
    try:
        most_recent_time = fetch_table_with_psycopg2(connection_Test, f"""
        SELECT MAX(datetime) 
        FROM strength_{signal};
        """).loc[0, 'max']
    except Exception as e:
        print(f"Error fetching MAX(datetime) for {signal}: {e}")
        most_recent_time = None

    return most_recent_time


most_recent_time_30m = get_timestamps('30m')
most_recent_time_4h = get_timestamps('4h')
most_recent_time_6h = get_timestamps('6h')
most_recent_time_1d = get_timestamps('1d')

# Evaluate which is the most old value between all tables to stablish as the minimum data to retrieve.
timestamp_condition = min(most_recent_time_30m, most_recent_time_4h,
                          most_recent_time_6h, most_recent_time_1d)
# Get 40 days before for WAD and EMA calculations.
timestamp_condition = timestamp_condition - 86400000*108

# List for token evaluation.
time_list = [('30m', most_recent_time_30m),
             ('4h', most_recent_time_4h),
             ('6h', most_recent_time_6h),
             ('1d', most_recent_time_1d)]

# Dataframes to save token's processed data.
df_wad_30m_A = pd.DataFrame()
df_wad_4h_A = pd.DataFrame()
df_wad_6h_A = pd.DataFrame()
df_wad_1d_A = pd.DataFrame()

df_wad_30m_B = pd.DataFrame()
df_wad_4h_B = pd.DataFrame()
df_wad_6h_B = pd.DataFrame()
df_wad_1d_B = pd.DataFrame()

df_ema_30m_A = pd.DataFrame()
df_ema_4h_A = pd.DataFrame()
df_ema_6h_A = pd.DataFrame()
df_ema_1d_A = pd.DataFrame()

df_ema_30m_B = pd.DataFrame()
df_ema_4h_B = pd.DataFrame()
df_ema_6h_B = pd.DataFrame()
df_ema_1d_B = pd.DataFrame()


def evaluate_per_tokens(tokens, group):
    """
    Evaluates and processes data for a list of tokens.

    This function performs the following operations for each token in the provided list:
    1. Fetches 30-minute closed candles data from the timestamp_condition.
    2. Preprocesses and cleans the 30-minute closed candles data.
    3. Extracts the latest live data for the token and appends it to the closed candles data.
    4. Filters the data by timestamps and resamples it if necessary.
    5. Fetches and preprocesses strength data for the token.
    6. Analyzes the WAD (Williams Accumulation/Distribution) for each token.
    7. Concatenates the results into a global dataframe based on the group (A or B).

    Args:
        tokens (list): A list of tokens to be evaluated.
        group (str): The group to which the tokens belong ('A' or 'B').

    Returns:
        None
    """
    global df_wad_30m_A, df_wad_4h_A, df_wad_6h_A, df_wad_1d_A, \
        df_wad_30m_B, df_wad_4h_B, df_wad_6h_B, df_wad_1d_B, \
        df_ema_30m_A, df_ema_4h_A, df_ema_6h_A, df_ema_1d_A, \
        df_ema_30m_B, df_ema_4h_B, df_ema_6h_B, df_ema_1d_B

    for token in tokens:

        # Fetch 30 minutes closed candles from the timestamp condition.
        sql_query_close = f"""
        SELECT * 
        FROM closed_candles 
        WHERE token = '{token}' 
        AND datetime >= {timestamp_condition};
        """

        df_closed_data = fetch_table_with_psycopg2(
            connection_Test, sql_query_close)

        # Preprocess 30 min closed candles data.
        df_closed_data = reformat_dates(df_closed_data)
        df_closed_data = sorting_by_columns(df_closed_data, 'datetime')

        # Clean duplicates in 30 min closed candles data.
        df_closed_data = remove_duplicates_from_df(df_closed_data)

        # Extract the last data in live data for the token.
        last_live_data = df_live_data[df_live_data['token'] == token]
        last_live_data = last_live_data[last_live_data['event_time']
                                        == last_live_data['event_time'].max()]

        # Add the last live data to the 30 min closed candles data.
        # If the last row from closed_data has the same datetime from the live_data it is replace by live data.
        if len(df_closed_data) > 0 and len(last_live_data) > 0:
            if df_closed_data.iloc[-1]['datetime'] == last_live_data.iloc[0]['datetime']:
                last_live_data = last_live_data.drop(last_live_data.index[0])

        df_closed_data_30m_for_wad = pd.concat(
            [df_closed_data, last_live_data], ignore_index=True)

        for df_signal, most_recent_time in time_list:
            #! WAD
            # Obtain the time of the previous row of the most recent time.
            if df_signal == '30m':
                most_recent_time_fix = most_recent_time - 1800000  # 30 min before
            if df_signal == '4h':
                most_recent_time_fix = most_recent_time - \
                    (3600000*4)  # 4 hours before
            if df_signal == '6h':
                most_recent_time_fix = most_recent_time - \
                    (3600000*6)  # 6 hours before
            if df_signal == '1d':
                most_recent_time_fix = most_recent_time - \
                    (86400000)  # 1 day before

            # Filter just the needed data in each signal.
            df_closed_data_for_wad = df_closed_data_30m_for_wad[df_closed_data_30m_for_wad['datetime'] >= pd.to_datetime(
                most_recent_time_fix, unit='ms', utc=True)].reset_index(drop=True)

            # Resample if needed.
            if df_signal != '30m':
                df_closed_data_for_wad = resample_candle_data(
                    df_closed_data_for_wad, df_signal)

            # Make sure is sorted.
            df_closed_data_for_wad.sort_values(by='datetime', inplace=True)

            # Bring the 58 most recent rows for strengh data.
            # Need at least 58 rows in order to calculate the sma57.
            sql_query_strength = f"""
            SELECT *
            FROM strength_{df_signal}
            WHERE token = '{token.upper()}' 
            ORDER BY datetime DESC
            LIMIT 58;"""

            df_strength_token = fetch_table_with_psycopg2(
                connection_Test, sql_query_strength)

            # Preprocess strength data.
            df_strength_token['datetime'] = pd.to_datetime(
                df_strength_token['datetime'], unit='ms', utc=True)
            df_strength_token = sorting_by_columns(
                df_strength_token, 'datetime')
            df_strength_token['strength'] = df_strength_token['strength'].astype(
                'float64')

            # Create the new strength dataframe by analyzing each token.
            df_wad_token = analyze_wad_for_each_token(
                df_closed_data_for_wad, df_strength_token)

            # Concatenate all tokens results in 1 dataframe.
            if group == 'A':
                if df_signal == '30m':
                    df_wad_30m_A = pd.concat(
                        [df_wad_30m_A, df_wad_token], ignore_index=True)
                elif df_signal == '4h':
                    df_wad_4h_A = pd.concat(
                        [df_wad_4h_A, df_wad_token], ignore_index=True)
                elif df_signal == '6h':
                    df_wad_6h_A = pd.concat(
                        [df_wad_6h_A, df_wad_token], ignore_index=True)
                elif df_signal == '1d':
                    df_wad_1d_A = pd.concat(
                        [df_wad_1d_A, df_wad_token], ignore_index=True)
            else:
                if df_signal == '30m':
                    df_wad_30m_B = pd.concat(
                        [df_wad_30m_B, df_wad_token], ignore_index=True)
                elif df_signal == '4h':
                    df_wad_4h_B = pd.concat(
                        [df_wad_4h_B, df_wad_token], ignore_index=True)
                elif df_signal == '6h':
                    df_wad_6h_B = pd.concat(
                        [df_wad_6h_B, df_wad_token], ignore_index=True)
                elif df_signal == '1d':
                    df_wad_1d_B = pd.concat(
                        [df_wad_1d_B, df_wad_token], ignore_index=True)

            #! EMA
            # Obtain the close data within 36 previous rows from the third previous row. (36 + 4)
            if df_signal == '30m':
                most_recent_time_ema = most_recent_time - 1800000*108  # 30 min before
            if df_signal == '4h':
                most_recent_time_ema = most_recent_time - \
                    (3600000*4)*108  # 4 hours before
            if df_signal == '6h':
                most_recent_time_ema = most_recent_time - \
                    (3600000*6)*108  # 6 hours before
            if df_signal == '1d':
                most_recent_time_ema = most_recent_time - \
                    (86400000)*108  # 1 day before

            # Filter just the needed data in each signal.
            df_ema_token = df_closed_data_30m_for_wad[df_closed_data_30m_for_wad['datetime'] >= pd.to_datetime(
                most_recent_time_ema, unit='ms', utc=True)].reset_index(drop=True)

            # Resample if needed.
            if df_signal != '30m':
                df_ema_token = resample_candle_data(df_ema_token, df_signal)

            # Make sure is sorted.
            df_ema_token.sort_values(by='datetime', inplace=True)

            # Calculate EMA for each token.
            df_ema_token['ema_36'] = ta.trend.ema_indicator(
                df_ema_token['close'], window=36)

            # Prepare data for classification.
            df_ema_token['close_prev'] = df_ema_token['close'].shift(1)
            df_ema_token['ema_36_prev'] = df_ema_token['ema_36'].shift(1)

            # Drop NaN values.
            df_ema_token.dropna(subset=['ema_36'], inplace=True)

            # Clasify EMA.
            df_ema_token['ema_36_class'] = 'Below EMA'
            df_ema_token.loc[df_ema_token['close'] >
                             df_ema_token['ema_36'], 'ema_36_class'] = 'Above EMA'
            df_ema_token.loc[(df_ema_token['close'] > df_ema_token['ema_36']) & (
                df_ema_token['close_prev'] <= df_ema_token['ema_36_prev']), 'ema_36_class'] = 'Just Crossed Above'
            df_ema_token.loc[(df_ema_token['close'] < df_ema_token['ema_36']) & (
                df_ema_token['close_prev'] >= df_ema_token['ema_36_prev']), 'ema_36_class'] = 'Just Crossed Below'

            # Drop unnecessary columns.
            df_ema_token.drop(
                columns=['close_prev', 'ema_36_prev'], inplace=True)

            # Calculate EMA distance.
            df_ema_token['ema_36_dist'] = (
                df_ema_token['close'] - df_ema_token['ema_36']).abs() / df_ema_token['ema_36']

            # Concatenate all tokens results in 1 dataframe.
            if group == 'A':
                if df_signal == '30m':
                    df_ema_30m_A = pd.concat(
                        [df_ema_30m_A, df_ema_token], ignore_index=True)
                elif df_signal == '4h':
                    df_ema_4h_A = pd.concat(
                        [df_ema_4h_A, df_ema_token], ignore_index=True)
                elif df_signal == '6h':
                    df_ema_6h_A = pd.concat(
                        [df_ema_6h_A, df_ema_token], ignore_index=True)
                elif df_signal == '1d':
                    df_ema_1d_A = pd.concat(
                        [df_ema_1d_A, df_ema_token], ignore_index=True)
            else:
                if df_signal == '30m':
                    df_ema_30m_B = pd.concat(
                        [df_ema_30m_B, df_ema_token], ignore_index=True)
                elif df_signal == '4h':
                    df_ema_4h_B = pd.concat(
                        [df_ema_4h_B, df_ema_token], ignore_index=True)
                elif df_signal == '6h':
                    df_ema_6h_B = pd.concat(
                        [df_ema_6h_B, df_ema_token], ignore_index=True)
                elif df_signal == '1d':
                    df_ema_1d_B = pd.concat(
                        [df_ema_1d_B, df_ema_token], ignore_index=True)

        logger.info(
            f'Process complete for {token}.\nComplete {(np.where(tokens == token)[0][0]+1) * 100 / len(tokens):.2f}%.')


# Describe the token threads groups (2 groups).
thread_A = threading.Thread(target=evaluate_per_tokens, args=[tokens_A, 'A'])
thread_B = threading.Thread(target=evaluate_per_tokens, args=[tokens_B, 'B'])

thread_A.start()  # Start the thread A
thread_B.start()  # Start the thread B

thread_A.join()  # Wait conclusion of thread A.
thread_B.join()  # Wait conclusion of thread B.

# Integrate both groups of tokens in one.
df_wad_30m = pd.concat([df_wad_30m_A, df_wad_30m_B], ignore_index=True)
df_wad_4h = pd.concat([df_wad_4h_A, df_wad_4h_B], ignore_index=True)
df_wad_6h = pd.concat([df_wad_6h_A, df_wad_6h_B], ignore_index=True)
df_wad_1d = pd.concat([df_wad_1d_A, df_wad_1d_B], ignore_index=True)

df_ema_30m = pd.concat([df_ema_30m_A, df_ema_30m_B], ignore_index=True)
df_ema_4h = pd.concat([df_ema_4h_A, df_ema_4h_B], ignore_index=True)
df_ema_6h = pd.concat([df_ema_6h_A, df_ema_6h_B], ignore_index=True)
df_ema_1d = pd.concat([df_ema_1d_A, df_ema_1d_B], ignore_index=True)

# Reformat token names to uppercase.
df_wad_30m = tokens_to_uppercase(df_wad_30m)
df_wad_4h = tokens_to_uppercase(df_wad_4h)
df_wad_6h = tokens_to_uppercase(df_wad_6h)
df_wad_1d = tokens_to_uppercase(df_wad_1d)

df_ema_30m = tokens_to_uppercase(df_ema_30m)
df_ema_4h = tokens_to_uppercase(df_ema_4h)
df_ema_6h = tokens_to_uppercase(df_ema_6h)
df_ema_1d = tokens_to_uppercase(df_ema_1d)

# %%
# Define the processes for next threads.
# Thread_1: save_strength_tables.
# Thread_2: continue_the_process.


def save_strength_tables(signal_list):
    """
    Save and update strength tables in the database for a list of signals.

    This function processes and updates the strength tables in the database for each signal in the provided list.
    It performs the following operations:
    1. Filters and preprocesses the strength data.
    2. Deletes existing rows in the strength table that match the most recent time.
    3. Updates the strength table with new data.
    4. Inserts new rows into the strength table.

    Args:
        signal_list (list): A list of tuples, where each tuple contains:
            - signal (str): The name of the signal.
            - most_recent_time (int): The most recent timestamp for the signal.
            - df_wad_signal (pd.DataFrame): A DataFrame containing the WAD signal data.

    Returns:
        None
    """
    # Stablish a connection using the pool.
    connection_Test_A = conn_pool.getconn()

    # ? FEED STRENGTH TABLES IN DATABASE.
    for signal, most_recent_time, df_wad_signal in signal_list:

        # Create a dataframe just with the necesary columns.
        strength_filtered = df_wad_signal[[
            'datetime', 'token', 'wad', 'sma57', 'sma25']]

        # Clean data
        strength_filtered['datetime'] = strength_filtered['datetime'].astype(
            'int64') // 1000000
        strength_filtered = strength_filtered.sort_values('datetime')
        strength_filtered.fillna(0)

        # Identify the live data.
        live_signal = strength_filtered.iloc[-1]

        # Get the dataframe for the data that needs to be updated.
        if most_recent_time == live_signal['datetime']:
            df_signal_extract = strength_filtered[strength_filtered['datetime']
                                                  == live_signal['datetime']]
        elif most_recent_time == None:
            df_signal_extract = pd.DataFrame()
        else:
            df_signal_extract = strength_filtered[strength_filtered['datetime']
                                                  >= most_recent_time]

        #! Rows to update.
        df_signal_extract_update = df_signal_extract[df_signal_extract['datetime']
                                                     == most_recent_time]

        # Delete old data.
        del_table = df_signal_extract_update[['datetime', 'token']]

        data_io_del = io.StringIO()
        del_table.to_csv(data_io_del, sep=";", index=False, header=False)
        data_io_del.seek(0)

        with connection_Test_A.cursor() as cursor_A:
            try:
                cursor_A.execute(
                    f"CREATE TEMP TABLE temp_delete_strength_{signal} (datetime BIGINT, token TEXT)")
                cursor_A.copy_from(
                    data_io_del, f"temp_delete_strength_{signal}", sep=";")

                cursor_A.execute(f"""
                    DELETE FROM strength_{signal}
                    USING temp_delete_strength_{signal}
                    WHERE strength_{signal}.datetime = temp_delete_strength_{signal}.datetime
                    AND strength_{signal}.token = temp_delete_strength_{signal}.token
                """)

                connection_Test_A.commit()
            except Exception as e:
                logger.info(f"Error in DELETE: {e}")
                connection_Test_A.rollback()

        # Replace data deleted with the updated version.
        data_io_upd = io.StringIO()
        df_signal_extract_update.to_csv(
            data_io_upd, sep=";", index=False, header=False)
        data_io_upd.seek(0)

        with connection_Test_A.cursor() as cursor_A:
            try:
                if data_io_upd.getvalue().strip():  # Verify there is data
                    cursor_A.copy_from(data_io_upd, f"strength_{signal}", sep=";", columns=(
                        'datetime', 'token', 'strength', 'sma_57', 'sma_25'))

                    connection_Test_A.commit()
                    logger.info(f'Strength table updated for {signal}')
                else:
                    logger.info("No data to update with COPY.")
            except Exception as e:
                logger.info(f"Error in UPDATE: {e}")
                connection_Test_A.rollback()

        #! Rows to insert.
        df_signal_extract_insert = df_signal_extract[~(
            df_signal_extract['datetime'] == most_recent_time)]

        data_io_ins = io.StringIO()
        df_signal_extract_insert.to_csv(
            data_io_ins, sep=";", index=False, header=False)
        data_io_ins.seek(0)

        with connection_Test_A.cursor() as cursor_A:
            try:
                cursor_A.copy_from(data_io_ins, f"strength_{signal}", sep=";", columns=(
                    'datetime', 'token', 'strength', 'sma_57', 'sma_25'))

                connection_Test_A.commit()
                logger.info(f'Strength table inserted for {signal}')
            except Exception as e:
                logger.info(f"Error: {e}")

        logger.info(f'Strength complete {signal}')

    # Close the connection of the thread and free memory.
    conn_pool.putconn(connection_Test_A)
    gc.collect()


def continue_the_process(signal_list):
    """
    Processes and updates signal and trade data, and generates plots.

    This function performs the following operations for each signal in the provided list:
    1. Updates the signal tables in the database.
    2. Updates the trade tables in the database.
    3. Generates and saves various plots based on the updated data.

    Args:
        signal_list (list): A list of tuples, where each tuple contains:
            - signal (str): The name of the signal.
            - most_recent_time (int): The most recent timestamp for the signal.
            - df_signal (pd.DataFrame): A DataFrame containing the signal data.

    Returns:
        None
    """

    descending_columns = ['strong_sell',
                          'medium_sell', 'first_call_sell', 'old_sell']

    def sort_tokens(row, df_ema, df_trades):

        datetime = row['DateTime']

        def get_sort_key(token, column):

            entry = df_ema[(df_ema['datetime'] == datetime)
                           & (df_ema['token'] == token)]
            if not entry.empty:
                classification_rank = class_order[entry['ema_36_class'].values[0]]
                ema_distance = entry['ema_36_dist'].values[0]
                classification_rank = - \
                    classification_rank if column in descending_columns else classification_rank
                return (classification_rank, ema_distance)
            else:
                # Si no hay datos, se pone al final y sin clasificación
                return (float('inf'), float('inf'), None)

        sorted_row = {}

        for col in df_trades.columns:
            if col != 'DateTime':
                sorted_tokens = sorted(
                    row[col], key=lambda token: get_sort_key(token, col))
                # Obtener la clasificación
                sorted_classes = [get_sort_key(
                    token, col)[0] for token in sorted_tokens]

                sorted_row[col] = sorted_tokens
                sorted_row[f"{col}_class"] = [abs(x) if x != float(
                    # Nueva columna con clasificaciones ordenadas
                    'inf') else 0 for x in sorted_classes]

        return sorted_row

    # First iteration file check.
    first_iteration_file = 'first_iteration_done.txt'
    first_iteration = not os.path.exists(first_iteration_file)

    # Updating state.
    update_date('Updating...')
    update_status(
        f"Service On: Updating new data for datetime: {df_live_data['event_time'].max().strftime('%Y-%m-%d %H:%M:%S')}")

    # Stablish a connection using the pool.
    connection_Test_B = conn_pool.getconn()

    for signal, most_recent_time, df_signal, df_ema in signal_list:

        # ? FEED SIGNAL TABLES IN DATABASE.

        # Transform date format for calculations.
        most_recent_time_dt = pd.to_datetime(
            most_recent_time, unit='ms', utc=True)

        # Get the live data row.
        df_signal = df_signal.sort_values('DateTime', ignore_index=True)
        live_signal = df_signal.iloc[-1]

        # New columns for plots.
        df_signal['Both_Below'] = df_signal.apply(lambda x: array_to_string(intersection_token_entries(
            string_to_array(x['Below_25']), string_to_array(x['Below_57']))), axis=1)
        df_signal['Both_Above'] = df_signal.apply(lambda x: array_to_string(intersection_token_entries(
            string_to_array(x['Above_25']), string_to_array(x['Above_57']))), axis=1)
        df_signal['Both_Cross_Down'] = df_signal.apply(lambda x: array_to_string(intersection_token_entries(
            string_to_array(x['Cross_Down_25']), string_to_array(x['Cross_Down_57']))), axis=1)
        df_signal['Both_Cross_Up'] = df_signal.apply(lambda x: array_to_string(intersection_token_entries(
            string_to_array(x['Cross_Up_25']), string_to_array(x['Cross_Up_57']))), axis=1)

        df_signal['Both_Below'] = df_signal['Both_Below'].apply(
            lambda x: np.nan if x == '' else x)
        df_signal['Both_Above'] = df_signal['Both_Above'].apply(
            lambda x: np.nan if x == '' else x)
        df_signal['Both_Cross_Down'] = df_signal['Both_Cross_Down'].apply(
            lambda x: np.nan if x == '' else x)
        df_signal['Both_Cross_Up'] = df_signal['Both_Cross_Up'].apply(
            lambda x: np.nan if x == '' else x)

        # Get the dataframe for the data that needs to be updated.
        if most_recent_time_dt == live_signal['DateTime']:
            df_signal_extract = df_signal[-2:]
        else:
            df_signal_extract = df_signal[df_signal['DateTime']
                                          >= most_recent_time_dt]

        df_signal_extract['DateTime'] = df_signal_extract['DateTime'].astype(
            'int64') // 1000000

        #! Rows to update.
        df_signal_extract_update = df_signal_extract[df_signal_extract['DateTime']
                                                     == most_recent_time]

        # Delete old data.
        del_list = df_signal_extract_update['DateTime'].unique().tolist()
        del_list_str = ', '.join(map(str, del_list))

        with connection_Test_B.cursor() as cursor:
            try:
                cursor.execute(f"""
                    DELETE FROM signal_{signal}
                    WHERE datetime IN ({del_list_str})            
                """)
                connection_Test_B.commit()
            except Exception as e:
                logger.info(f"Error en DELETE: {e}")
                connection_Test_B.rollback()

        # Replace data deleted with the updated version.
        data_io_upd = io.StringIO()
        df_signal_extract_update.to_csv(
            data_io_upd, sep=";", index=False, header=False)
        data_io_upd.seek(0)

        with connection_Test_B.cursor() as cursor:
            try:
                cursor.copy_from(data_io_upd, f"signal_{signal}", sep=";", columns=('datetime', 'below_25', 'above_25', 'cross_down_25', 'cross_up_25',
                                 'below_57', 'above_57', 'cross_down_57', 'cross_up_57', 'both_below', 'both_above', 'both_cross_down', 'both_cross_up'))
                connection_Test_B.commit()
                logger.info(f'Signal table updated for {signal}')
            except Exception as e:
                logger.info(f"Error en UPDATE: {e}")
                connection_Test_B.rollback()

        #! Rows to insert.
        df_signal_extract_insert = df_signal_extract[~(
            df_signal_extract['DateTime'] == most_recent_time)]

        data_io_ins = io.StringIO()
        df_signal_extract_insert.to_csv(
            data_io_ins, sep=";", index=False, header=False)
        data_io_ins.seek(0)

        with connection_Test_B.cursor() as cursor:
            try:
                cursor.copy_from(data_io_ins, f"signal_{signal}", sep=";", columns=('datetime', 'below_25', 'above_25', 'cross_down_25', 'cross_up_25',
                                 'below_57', 'above_57', 'cross_down_57', 'cross_up_57', 'both_below', 'both_above', 'both_cross_down', 'both_cross_up'))
                connection_Test_B.commit()
                logger.info(f'Signal table inserted for {signal}')
            except Exception as e:
                logger.info(f"Error en INSERT: {e}")
                connection_Test_B.rollback()

        logger.info(f'Signal table completed for {signal}')

        # ? FEED TRADES TABLES IN DATABASE.

        # Find trades opportunities for each new signal row.
        values = []
        for _, row in df_signal_extract.iterrows():
            row_long_trade_data, row_short_trade_data = find_trades_flags(row)
            value = {
                'DateTime': row['DateTime'],
                'strong_buy': array_to_string(row_long_trade_data[3]),
                'medium_buy': array_to_string(row_long_trade_data[2]),
                'first_call_buy': array_to_string(row_long_trade_data[1]),
                'old_buy': array_to_string(row_long_trade_data[0]),
                'strong_sell': array_to_string(row_short_trade_data[3]),
                'medium_sell': array_to_string(row_short_trade_data[2]),
                'first_call_sell': array_to_string(row_short_trade_data[1]),
                'old_sell': array_to_string(row_short_trade_data[0]),
                'new_datetime': pd.to_datetime(row['DateTime'], unit='ms', utc=True)
            }
            value = {k: ('NaN' if v == '' else v) for k, v in value.items()}
            values.append(value)
        df_values = pd.DataFrame(values)

        #! Rows to update.
        df_signal_extract_update = df_values[df_values['DateTime']
                                             == most_recent_time]

        # Delete old data.
        del_list = df_signal_extract_update['DateTime'].unique().tolist()
        del_list_str = ', '.join(map(str, del_list))

        with connection_Test_B.cursor() as cursor:
            try:
                cursor.execute(f"""
                    DELETE FROM trades_data_{signal}
                    WHERE datetime IN ({del_list_str})            
                """)
                connection_Test_B.commit()
            except Exception as e:
                logger.info(f"Error en DELETE: {e}")
                connection_Test_B.rollback()

        # Replace data deleted with the updated version.
        data_io_upd = io.StringIO()
        df_signal_extract_update.to_csv(
            data_io_upd, sep=";", index=False, header=False)
        data_io_upd.seek(0)

        with connection_Test_B.cursor() as cursor:
            try:
                cursor.copy_from(data_io_upd, f"trades_data_{signal}", sep=";", columns=(
                    'datetime', 'strong_buy', 'medium_buy', 'first_call_buy', 'old_buy', 'strong_sell', 'medium_sell', 'first_call_sell', 'old_sell', 'new_datetime'))
                connection_Test_B.commit()
                logger.info(f'Trades table updated for {signal}')
            except Exception as e:
                logger.info(f"Error en UPDATE: {e}")
                connection_Test_B.rollback()

        #! Rows to insert.
        df_signal_extract_insert = df_values[~(
            df_values['DateTime'] == most_recent_time)]

        data_io_ins = io.StringIO()
        df_signal_extract_insert.to_csv(
            data_io_ins, sep=";", index=False, header=False)
        data_io_ins.seek(0)

        with connection_Test_B.cursor() as cursor:
            try:
                cursor.copy_from(data_io_ins, f"trades_data_{signal}", sep=";", columns=(
                    'datetime', 'strong_buy', 'medium_buy', 'first_call_buy', 'old_buy', 'strong_sell', 'medium_sell', 'first_call_sell', 'old_sell', 'new_datetime'))
                connection_Test_B.commit()
                logger.info(f'Trades table inserted for {signal}')
            except Exception as e:
                logger.info(f"Error en INSERT: {e}")
                connection_Test_B.rollback()

        logger.info(f'Trades table completed for {signal}')

        # ? CREATING TREEPLOTS.

        # Get the last 4 rows to plot tree maps.
        sql_query_signal = f"""
        SELECT *
        FROM trades_data_{signal}         
        ORDER BY datetime DESC
        LIMIT 4;"""

        df_trades = fetch_table_with_psycopg2(
            connection_Test, sql_query_signal)

        # Preprocess trades data.
        df_trades['datetime'] = pd.to_datetime(
            df_trades['datetime'], unit='ms', utc=True)
        df_trades = sorting_by_columns(df_trades, 'datetime')

        df_trades_fix = df_trades.drop(columns=['datetime', 'new_datetime']).map(
            lambda x: [] if x == 'NaN' else string_to_array(x))
        df_trades_fix.insert(0, 'DateTime', df_trades['datetime'])

        # Ratio tokens in each section.
        trades_counts = df_trades.drop(columns=['datetime', 'new_datetime']).map(
            lambda x: f'(0/{len_tokens} - 0%)' if x == 'NaN' else f'({len(string_to_array(x))}/{len_tokens} - {len(string_to_array(x))*100/len_tokens:.2f}%)')
        trades_counts.columns = [
            col + '_ratio' for col in trades_counts.columns]
        trades_counts.insert(0, 'DateTime', df_trades['datetime'])

        # Aplicar la función de ordenamiento a cada fila
        sorted_results = df_trades_fix.apply(
            lambda row: sort_tokens(row, df_ema, df_trades_fix), axis=1)

        # Agregar las columnas ordenadas al DataFrame original
        for col in df_trades_fix.columns:
            if col != 'DateTime':
                df_trades_fix[col] = sorted_results.apply(lambda x: x[col])
                df_trades_fix[f"{col}_class"] = sorted_results.apply(
                    lambda x: x[f"{col}_class"])

        # Merge counting df and data df
        df_trades_fix = df_trades_fix.merge(
            trades_counts, on='DateTime', how='left', sort=True)

        # Identify the live row and the 3 previous rows.
        live_signal = df_trades_fix.iloc[-1]
        one_previous_signal = df_trades_fix.iloc[-2]
        two_previous_signal = df_trades_fix.iloc[-3]
        three_previous_signal = df_trades_fix.iloc[-4]

        #! LIVE DATA
        live_long_trade_data = {'old_buy': {'data': live_signal['old_buy'], 'ratio': live_signal['old_buy_ratio'], 'classes': live_signal['old_buy_class']},
                                'first_call_buy': {'data': live_signal['first_call_buy'], 'ratio': live_signal['first_call_buy_ratio'], 'classes': live_signal['first_call_buy_class']},
                                'medium_buy': {'data': live_signal['medium_buy'], 'ratio': live_signal['medium_buy_ratio'], 'classes': live_signal['medium_buy_class']},
                                'strong_buy': {'data': live_signal['strong_buy'], 'ratio': live_signal['strong_buy_ratio'], 'classes': live_signal['strong_buy_class']}
                                }

        live_short_trade_data = {'old_sell': {'data': live_signal['old_sell'], 'ratio': live_signal['old_sell_ratio'], 'classes': live_signal['old_sell_class']},
                                 'first_call_sell': {'data': live_signal['first_call_sell'], 'ratio': live_signal['first_call_sell_ratio'], 'classes': live_signal['first_call_sell_class']},
                                 'medium_sell': {'data': live_signal['medium_sell'], 'ratio': live_signal['medium_sell_ratio'], 'classes': live_signal['medium_sell_class']},
                                 'strong_sell': {'data': live_signal['strong_sell'], 'ratio': live_signal['strong_sell_ratio'], 'classes': live_signal['strong_sell_class']}
                                 }

        #! ONE PREVIOUS ROW DATA
        one_previous_long_trade_data = {'old_buy': {'data': one_previous_signal['old_buy'], 'ratio': one_previous_signal['old_buy_ratio'], 'classes': one_previous_signal['old_buy_class']},
                                        'first_call_buy': {'data': one_previous_signal['first_call_buy'], 'ratio': one_previous_signal['first_call_buy_ratio'], 'classes': one_previous_signal['first_call_buy_class']},
                                        'medium_buy': {'data': one_previous_signal['medium_buy'], 'ratio': one_previous_signal['medium_buy_ratio'], 'classes': one_previous_signal['medium_buy_class']},
                                        'strong_buy': {'data': one_previous_signal['strong_buy'], 'ratio': one_previous_signal['strong_buy_ratio'], 'classes': one_previous_signal['strong_buy_class']}
                                        }

        one_previous_short_trade_data = {'old_sell': {'data': one_previous_signal['old_sell'], 'ratio': one_previous_signal['old_sell_ratio'], 'classes': one_previous_signal['old_sell_class']},
                                         'first_call_sell': {'data': one_previous_signal['first_call_sell'], 'ratio': one_previous_signal['first_call_sell_ratio'], 'classes': one_previous_signal['first_call_sell_class']},
                                         'medium_sell': {'data': one_previous_signal['medium_sell'], 'ratio': one_previous_signal['medium_sell_ratio'], 'classes': one_previous_signal['medium_sell_class']},
                                         'strong_sell': {'data': one_previous_signal['strong_sell'], 'ratio': one_previous_signal['strong_sell_ratio'], 'classes': one_previous_signal['strong_sell_class']}
                                         }

        #! TWO PREVIOUS ROW DATA
        two_previous_long_trade_data = {'old_buy': {'data': two_previous_signal['old_buy'], 'ratio': two_previous_signal['old_buy_ratio'], 'classes': two_previous_signal['old_buy_class']},
                                        'first_call_buy': {'data': two_previous_signal['first_call_buy'], 'ratio': two_previous_signal['first_call_buy_ratio'], 'classes': two_previous_signal['first_call_buy_class']},
                                        'medium_buy': {'data': two_previous_signal['medium_buy'], 'ratio': two_previous_signal['medium_buy_ratio'], 'classes': two_previous_signal['medium_buy_class']},
                                        'strong_buy': {'data': two_previous_signal['strong_buy'], 'ratio': two_previous_signal['strong_buy_ratio'], 'classes': two_previous_signal['strong_buy_class']}
                                        }

        two_previous_short_trade_data = {'old_sell': {'data': two_previous_signal['old_sell'], 'ratio': two_previous_signal['old_sell_ratio'], 'classes': two_previous_signal['old_sell_class']},
                                         'first_call_sell': {'data': two_previous_signal['first_call_sell'], 'ratio': two_previous_signal['first_call_sell_ratio'], 'classes': two_previous_signal['first_call_sell_class']},
                                         'medium_sell': {'data': two_previous_signal['medium_sell'], 'ratio': two_previous_signal['medium_sell_ratio'], 'classes': two_previous_signal['medium_sell_class']},
                                         'strong_sell': {'data': two_previous_signal['strong_sell'], 'ratio': two_previous_signal['strong_sell_ratio'], 'classes': two_previous_signal['strong_sell_class']}
                                         }

        #! THREE PREVIOUS ROW DATA
        three_previous_long_trade_data = {'old_buy': {'data': three_previous_signal['old_buy'], 'ratio': three_previous_signal['old_buy_ratio'], 'classes': three_previous_signal['old_buy_class']},
                                          'first_call_buy': {'data': three_previous_signal['first_call_buy'], 'ratio': three_previous_signal['first_call_buy_ratio'], 'classes': three_previous_signal['first_call_buy_class']},
                                          'medium_buy': {'data': three_previous_signal['medium_buy'], 'ratio': three_previous_signal['medium_buy_ratio'], 'classes': three_previous_signal['medium_buy_class']},
                                          'strong_buy': {'data': three_previous_signal['strong_buy'], 'ratio': three_previous_signal['strong_buy_ratio'], 'classes': three_previous_signal['strong_buy_class']}
                                          }

        three_previous_short_trade_data = {'old_sell': {'data': three_previous_signal['old_sell'], 'ratio': three_previous_signal['old_sell_ratio'], 'classes': three_previous_signal['old_sell_class']},
                                           'first_call_sell': {'data': three_previous_signal['first_call_sell'], 'ratio': three_previous_signal['first_call_sell_ratio'], 'classes': three_previous_signal['first_call_sell_class']},
                                           'medium_sell': {'data': three_previous_signal['medium_sell'], 'ratio': three_previous_signal['medium_sell_ratio'], 'classes': three_previous_signal['medium_sell_class']},
                                           'strong_sell': {'data': three_previous_signal['strong_sell'], 'ratio': three_previous_signal['strong_sell_ratio'], 'classes': three_previous_signal['strong_sell_class']}
                                           }

        # Execute plotting for rows feeded with the trades opportunities data.
        # Plotting optimization already applied, just the data that change is plotted.
        if len(df_signal_extract_insert) == 0 and not first_iteration:

            with open(fr'{cwd}/plots/{signal}_live_long_trade.json', "w") as file:
                json.dump(live_long_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_live_short_trade.json', "w") as file:
                json.dump(live_short_trade_data, file, indent=4)
        else:

            with open(fr'{cwd}/plots/{signal}_live_long_trade.json', "w") as file:
                json.dump(live_long_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_live_short_trade.json', "w") as file:
                json.dump(live_short_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_one_long_trade.json', "w") as file:
                json.dump(one_previous_long_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_one_short_trade.json', "w") as file:
                json.dump(one_previous_short_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_two_long_trade.json', "w") as file:
                json.dump(two_previous_long_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_two_short_trade.json', "w") as file:
                json.dump(two_previous_short_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_three_long_trade.json', "w") as file:
                json.dump(three_previous_long_trade_data, file, indent=4)

            with open(fr'{cwd}/plots/{signal}_three_short_trade.json', "w") as file:
                json.dump(three_previous_short_trade_data, file, indent=4)

            # Update titles.
            update_titles({signal: [one_previous_signal['DateTime'],
                          two_previous_signal['DateTime'], three_previous_signal['DateTime']]})

            # Create a temporal file to check first iteration.
            if first_iteration:
                with open(first_iteration_file, 'w') as f:
                    f.write('done')

        logger.info(f'Tree plots builded for {signal}')

        # ? CREATING LINEPLOTS.

        #  Select just the last 10 days for 30m data and 4 month for the other signals.
        if signal == '30m':
            timestamp_condition = int(
                (datetime.now() - relativedelta(days=10)).timestamp() * 1000)
        else:
            timestamp_condition = int(
                (datetime.now() - relativedelta(months=4)).timestamp() * 1000)

        # Bring the data since the time stamp.
        sql_query_signal = f"""
        SELECT * FROM signal_{signal}
        WHERE datetime >= {timestamp_condition};"""

        df_complete_signal = fetch_table_with_psycopg2(
            connection_Test, sql_query_signal)

        # Preprocess complete_signal data.
        df_complete_signal['datetime'] = pd.to_datetime(
            df_complete_signal['datetime'], unit='ms', utc=True)
        df_complete_signal = sorting_by_columns(df_complete_signal, 'datetime')

        signal_counts = df_complete_signal.drop(columns=['datetime']).map(
            lambda x: 0 if pd.isna(x) else len(string_to_array(x)))
        signal_counts.insert(0, 'DateTime', df_complete_signal['datetime'])

        # Create the plots and post them.
        for column in signal_counts.columns[1:]:
            if column in ['above_25', 'cross_up_25', 'above_57', 'cross_up_57', 'both_above', 'both_cross_up']:
                color = '#28b463'  # Green
            else:
                color = '#cb4335'  # Red

            # Define the range for the last 5 days
            last_date = signal_counts['DateTime'].iloc[-1] + \
                pd.Timedelta(days=1)
            start_date = signal_counts['DateTime'].iloc[-1] - \
                pd.Timedelta(days=5)

            # Create the line plot.
            fig = px.line(signal_counts, x='DateTime', y=column, title=column.replace(
                '_', ' ').title(), color_discrete_sequence=[color])
            fig.update_layout(
                title={'x': 0.5, 'xanchor': 'center'},
                yaxis_title='Number of tokens',
                xaxis_title='Date Time',
                xaxis=dict(tickangle=45, range=[start_date, last_date]),
                template='plotly_white',
                width=800,
                height=600
            )

            # Get the last value of the column
            last_value = signal_counts[column].iloc[-1]
            last_date = signal_counts['DateTime'].iloc[-1]

            # Add annotation with the last value
            fig.add_annotation(
                x=last_date,
                y=last_value,
                text=f'Current tokens: {last_value}',
                showarrow=True,
                arrowhead=1,
                ax=80,
                ay=-30,
                font=dict(
                    size=12,
                    color="black",
                    family="Arial"
                )
            )

            # Convert the figure to JSON
            plot = fig.to_json()

            # Save the plots to a JSON file
            with open(f'lineplots/{signal}_{column}.json', 'w') as f:
                json.dump(plot, f)

        logger.info(f'Line plots builded for {signal}')

    # Close the connection of the thread.
    conn_pool.putconn(connection_Test_B)


# %%
#! Coordinate the rest of the process's tasks.
# List for strengths
strenghts_list = [('30m', most_recent_time_30m, df_wad_30m),
                  ('4h', most_recent_time_4h, df_wad_4h),
                  ('6h', most_recent_time_6h, df_wad_6h),
                  ('1d', most_recent_time_1d, df_wad_1d)]

# Iterate over the condition columns and generate a complete dataframe for signal.
df_signal_30m = generate_signal_df(df_wad_30m, '30min')
df_signal_4h = generate_signal_df(df_wad_4h, '4h')
df_signal_6h = generate_signal_df(df_wad_6h, '6h')
df_signal_1d = generate_signal_df(df_wad_1d, '1d')

# List for signals.
signal_list = [('30m', most_recent_time_30m, df_signal_30m, df_ema_30m),
               ('4h', most_recent_time_4h, df_signal_4h, df_ema_4h),
               ('6h', most_recent_time_6h, df_signal_6h, df_ema_6h),
               ('1d', most_recent_time_1d, df_signal_1d, df_ema_1d)]

# Free some memory.
gc.collect()

# Define the threading process in two.
# postgresql_thread: will save in database tables all strength data generated.
# main thread: will continue to complete the whole process.
postgresql_thread = threading.Thread(
    target=save_strength_tables, args=[strenghts_list])

postgresql_thread.start()  # Start the postgresql thread

continue_the_process(signal_list)  # Continue with the main thread

postgresql_thread.join()  # Wait the postgresql thread to finish

# Update the time and status.
update_date(df_live_data['event_time'].max().strftime("%Y-%m-%d %H:%M:%S"))
update_status('Service On: Update completed')

# %%
# Finish the process and close conections.
conn_pool.putconn(connection_Test)
# %%
