import os
import copy
import pickle
import urllib
import zipfile
from datetime import *
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import stockstats
import talib
import tushare as ts
from tqdm import tqdm
import time

class DataProcessor:
    def __init__(
        self,
        token: str,
        ticker_list: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        adj = None,
    ):
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe = pd.DataFrame()
        self.adj = adj
        ts.set_token(token)

    def download_data(self, ticker_list = None):
        if ticker_list is not None:
            self.ticker_list = ticker_list
        for i in tqdm(self.ticker_list, total=len(self.ticker_list)):
            df_temp = ts.pro_bar(ts_code = i, start_date = self.start_date, end_date = self.end_date, adj = self.adj)
            self.dataframe = self.dataframe.append(df_temp)
            time.sleep(0.1)

        self.dataframe.columns = [
                "tic",
                "date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "volume",
                "amount",
        ]
        self.dataframe.sort_values(by=["date", "tic"],inplace=True)
        print("Shape of DataFrame: ", self.dataframe.shape)

    def clean_data(self):
        dfc = copy.deepcopy(self.dataframe)

        dfcode = pd.DataFrame(columns=["tic"])
        dfdate = pd.DataFrame(columns=["date"])

        dfcode.tic = dfc.tic.unique()

        if "time" in dfc.columns.values.tolist():
            dfc = dfc.rename(columns={"time": "date"})

        dfdate.date = dfc.date.unique()
        dfdate.sort_values(by="date", ascending=False, ignore_index=True, inplace=True)

        # the old pandas may not support pd.merge(how="cross")
        try:
            df1 = pd.merge(dfcode, dfdate, how="cross")
        except:
            print("Please wait for a few seconds...")
            df1 = pd.DataFrame(columns=["tic", "date"])
            for i in range(dfcode.shape[0]):
                for j in range(dfdate.shape[0]):
                    df1 = df1.append(
                        pd.DataFrame(
                            data={
                                "tic": dfcode.iat[i, 0],
                                "date": dfdate.iat[j, 0],
                            },
                            index=[(i + 1) * (j + 1) - 1],
                        )
                    )

        df2 = pd.merge(df1, dfc, how="left", on=["tic", "date"])

        # back fill missing data then front fill
        df3 = pd.DataFrame(columns=df2.columns)
        for i in self.ticker_list:
            df4 = df2[df2.tic == i].fillna(method="bfill").fillna(method="ffill")
            df3 = pd.concat([df3, df4], ignore_index=True)

        df3 = df3.fillna(0)

        # reshape dataframe
        df3 = df3.sort_values(by=["date", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", df3.shape)

        self.dataframe = df3


    def add_technical_indicator(
        self, tech_indicator_list: List[str], select_stockstats_talib: int = 0
    ):
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)


        self.dataframe.reset_index(drop=False, inplace=True)
        if "level_1" in self.dataframe.columns:
            self.dataframe.drop(columns=["level_1"], inplace=True)
        if "level_0" in self.dataframe.columns and "tic" not in self.dataframe.columns:
            self.dataframe.rename(columns={"level_0": "tic"}, inplace=True)
        assert select_stockstats_talib in {0, 1}
        print("tech_indicator_list: ", tech_indicator_list)
        if select_stockstats_talib == 0:  # use stockstats
            stock = stockstats.StockDataFrame.retype(self.dataframe)
            #print(help(stock))
            unique_ticker = stock.tic.unique()
            for indicator in tech_indicator_list:
                print("indicator: ", indicator)
                indicator_df = pd.DataFrame()
                for i in range(len(unique_ticker)):
                    try:
                        temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                        #print(temp_indicator)
                        temp_indicator = pd.DataFrame(temp_indicator)
                        temp_indicator["tic"] = unique_ticker[i]
                        temp_indicator["time"] = self.dataframe[
                            self.dataframe.tic == unique_ticker[i]
                        ]["time"].to_list()
                        indicator_df = pd.concat(
                            [indicator_df, temp_indicator],
                            axis=0,
                            join="outer",
                            ignore_index=True,
                        )
                    except Exception as e:
                        print(e)
                if not indicator_df.empty:
                    self.dataframe = self.dataframe.merge(
                        indicator_df[["tic", "time", indicator]],
                        on=["tic", "time"],
                        how="left",
                    )
        else:  # use talib
            final_df = pd.DataFrame()
            print("-------talib start------")
            for i in self.dataframe.tic.unique():
                tic_df = self.dataframe[self.dataframe.tic == i]
                print(tic_df)
                print(talib.MACD(
                    tic_df["close"],
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9,
                ))
                (
                    tic_df.loc["macd"],
                    tic_df.loc["macd_signal"],
                    tic_df.loc["macd_hist"],
                ) = talib.MACD(
                    tic_df["close"],
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9,
                )
                print(tic_df)
                tic_df.loc["rsi"] = talib.RSI(tic_df["close"], timeperiod=14)
                tic_df.loc["cci"] = talib.CCI(
                    tic_df["high"],
                    tic_df["low"],
                    tic_df["close"],
                    timeperiod=14,
                )
                tic_df.loc["dx"] = talib.DX(
                    tic_df["high"],
                    tic_df["low"],
                    tic_df["close"],
                    timeperiod=14,
                )
                final_df = pd.concat([final_df, tic_df], axis=0, join="outer")
            self.dataframe = final_df

        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        time_to_drop = self.dataframe[self.dataframe.isna().any(axis=1)].time.unique()
        self.dataframe = self.dataframe[~self.dataframe.time.isin(time_to_drop)]
        print("Succesfully add technical indicators")


    def add_turbulence(self):
        pass

    def add_vix(self):
        pass

    def df_to_array(self, if_vix: bool) -> np.array:
        pass

    def data_split(self, df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def run(
        self,
        ticker_list: str,
        technical_indicator_list: List[str],
        if_vix: bool,
        cache: bool = False,
        select_stockstats_talib: int = 0,
    ):
        pass


if __name__ == "__main__":
    ticker_list = ["600016.SH"]
    TRAIN_START_DATE = '2012-08-01'
    TRAIN_END_DATE= '2022-11-29'
    TRADE_START_DATE = '2022-08-01'
    TRADE_END_DATE = '2022-11-29'
    token0="27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"
    dp = DataProcessor(token=token0,start_date=TRAIN_START_DATE,end_date=TRAIN_END_DATE,time_interval="1d",ticker_list = ticker_list)
    dp.download_data(ticker_list)
    dp.clean_data()
    INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    #"close_30_sma",
    #"close_60_sma",
    "close_4_sma",
    "close_14_sma",
    ]
    #print(dp.dataframe)
    #print(dp.dataframe.open)
    dp.add_technical_indicator(INDICATORS,0)
    #print(dp.dataframe)
    #print(dp.dataframe.macd)
    dp.clean_data()
    train = dp.data_split(dp.dataframe, TRAIN_START_DATE, TRAIN_END_DATE) 
    print(f"len(train.tic.unique()): {len(train.tic.unique())}")
    print(train.shape)
    #print(f"train.head(): {train.head(10)}")
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(INDICATORS) + 2) + 1
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    from env_stocktrading_China_A_shares import StockTradingEnv
    env_kwargs = { "stock_dim": stock_dimension, "hmax": 1000, "initial_amount": 1000000, "buy_cost_pct": 6.87e-5, "sell_cost_pct": 1.0687e-3, "reward_scaling": 1e-4, "state_space": state_space, "action_space": stock_dimension, "tech_indicator_list": INDICATORS, "print_verbosity": 1, "initial_buy": False, "hundred_each_trade": True }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    #e_train_gym.reset()

    print("------ state run -------")
    print(e_train_gym.initial_amount)
    print(e_train_gym.state)
    print("------- step 1 ------")
    state,reward,terminal,_=e_train_gym.step(np.array([4]))
    print(state)
    print(type(state),terminal,reward)
    
    print("------- step 2 ------")
    state,reward,terminal,_=e_train_gym.step(np.array([40]))
    print(state)
    print(type(state),terminal,reward)
    print("------- step 3 ------")
    state,reward,terminal,_=e_train_gym.step(np.array([1000]))
    print(state)
    print(type(state),terminal,reward)
    print("------- step 4 ------")
    state,reward,terminal,_=e_train_gym.step(np.array([-100]))
    print(state)
    print(type(state),terminal,reward)
    
    

