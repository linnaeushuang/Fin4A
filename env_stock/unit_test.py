from data_processor import DataProcessor
from env_stocktrading_China_A_shares import StockTradingEnv
import numpy as np



def test_one_episode():
    ticker_list = ["600036.SH"]
    TRAIN_START_DATE = '2021-11-01'
    TRAIN_END_DATE= '2023-01-11'
    TRADE_START_DATE = '2021-01-01'
    TRADE_END_DATE = '2022-05-09'
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
    "close_4_sma",
    "close_14_sma",
    ]
    dp.add_technical_indicator(INDICATORS,0)
    dp.clean_data()
    print(dp.dataframe.tail()["close"])
    #train = dp.data_split(dp.dataframe, TRADE_START_DATE, TRADE_END_DATE) 
    #print(train.shape)
    #print(f"len(train.tic.unique()): {len(train.tic.unique())}")
    #print(train.shape)
    #stock_dimension = len(train.tic.unique())
    stock_dimension = len(dp.dataframe.tic.unique())
    # state space is [curr_amount,vol_1,close_1,n_techindex_1,vol_2,close_2,n_techindex_2,......]
    # so dim of state space is 1+(techindex_len+2)*stock_num
    state_space = stock_dimension * (len(INDICATORS) + 2) + 1
    env_kwargs = { "stock_dim": stock_dimension, "hmax": 1000, "initial_amount": 1000000, "buy_cost_pct": 6.87e-5, "sell_cost_pct": 1.0687e-3, "reward_scaling": 1e-4, "state_space": state_space, "action_space": stock_dimension, "tech_indicator_list": INDICATORS, "print_verbosity": 1, "initial_buy": False, "hundred_each_trade": True }
    #print(train.shape)
    e_train_gym = StockTradingEnv(df=dp.dataframe, **env_kwargs)
    terminal = False
    action = np.random.randn(stock_dimension) * 5
    while not terminal:
        action = np.random.randn(stock_dimension) * 5
        state, reward, terminal, _ = e_train_gym.step(action)
        """
        if state[0,2] > state[1,2]:
            action=np.array([50])
        else:
            action=np.array([-50])
        """



if __name__ == "__main__":
    test_one_episode()


