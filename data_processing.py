"""
Data Processing Module. This modules processes and norms orderbook L2 data that has been retrieved by the client.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime


class DataProcessor:
    """
    processes raw orderbook for Hawkes calibration.
    """
    
    def __init__(self):
        self.trades_df = None
        self.processed_events = None
        
    def load_and_process_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        load + clean and process L2 data.
    
        """
        df = trades_df.copy() #copy to avoid altering original in mem
        
        #  converting to correct type in case they aren't alreadt 
        df["time"] = pd.to_datetime(df["time"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        
        # leading with missing values such as NAN 
        df = df.dropna(subset=["price", "size"])
        
        # sorting by time here 
        df = df.sort_values("time").reset_index(drop=True)
        
        # time in secos from the init time 
        df["time_seconds"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
        
        self.trades_df = df
        return df
    
    def classify_trades(self, trades_df: pd.DataFrame, method: str = "side") -> pd.DataFrame:
        """
        trade classification as either buy  or sell.
        """
        df = trades_df.copy() #preventing altering orig data 
        
        if method == "side": # => using side column straight away since coinbase provides it 
    
            df["classified_side"] = df["side"]
        
        elif method == "tick":
            # is thats tick => compare that to previous price
            df["price_change"] = df["price"].diff()
            df["classified_side"] = "buy"  # thats the default to buy
            # else => sell 
            df.loc[df["price_change"] < 0, "classified_side"] = "sell"
            # Keep first trade without alterations if we hace info 
            if "side" in df.columns:
                df.loc[0, "classified_side"] = df.loc[0, "side"]
        
        return df
    
    def extract_hawkes_events(
        self, 
        trades_df: pd.DataFrame,
        max_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        precise event (times and types) for hawkes processes.
        """
        df = trades_df.copy()
        
        if "time_seconds" not in df.columns:
            df["time_seconds"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
        # if we don't got classified side then => classify 
        if "classified_side" not in df.columns:
            df = self.classify_trades(df)
        
        # filters by mac durations  
        if max_duration:
            df = df[df["time_seconds"] <= max_duration]
        
        # retrieving event time + types 
        event_times = df["time_seconds"].values
        event_types = (df["classified_side"] == "buy").astype(int).values
        # storeing them to ref them later 
        self.processed_events = {
            "times": event_times,
            "types": event_types,
            "duration": event_times[-1] if len(event_times) > 0 else 0
        }
        
        return event_times, event_types
    
    def compute_trade_intensity(
        self, 
        trades_df: pd.DataFrame, 
        window_seconds: float = 60.0
    ) -> pd.DataFrame:
        """
        Compute rolling (a rolling widow of) trade intensity (trades/second) here.
        """
        df = trades_df.copy() #3 copyig again to prev altering data 
        # calculating times in secs 
        if "time_seconds" not in df.columns:
            df["time_seconds"] = (df["time"] - df["time"].iloc[0]
                                  ).dt.total_seconds() # th e times in secs 
        
        # we are creating the trading bins here 
        max_time = df["time_seconds"].max()
        bins = np.arange(0, max_time + window_seconds, window_seconds) # based on the max time range
        
        # and calcing event  time below  
        df["time_bin"] = pd.cut(df["time_seconds"], bins=bins)
        intensity = df.groupby("time_bin").size() /window_seconds
        
        return intensity.reset_index(name="intensity") #resetting idx 
    
    def compute_ofi_from_orderbook(
        self, 
        orderbook_snapshots: List[Dict],
        levels: int = 5
    ) -> pd.DataFrame:
        """
        This method computes Order Flow Imbalance microstrcture OFI from the snapshots retrieved from the orderbook.
        
        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        """
        ofi_data = []
        
        #checking the snapshots 
        for snapshot in orderbook_snapshots:
            timestamp = snapshot["timestamp"] #retrieving the timestamp 
            bids = snapshot["bids"][:levels] #bids 
            asks = snapshot["asks"][:levels] #asks here 
            
            # alculating  aggregate  volume for bot bid and ask 
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            
            total_volume =bid_volume +ask_volume
            #if the tota volume is bigger than 0 then calc OFI otherwise => set to 0
            ofi = (bid_volume -ask_volume) / total_volume if total_volume > 0 else 0
            
            # calculating the mid price microstructurre 
            if bids and asks: #we need both available to calculate it
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            else:
                mid_price = None #if te condition isnt met => set to None 
            
            # update the data hashmap 
            ofi_data.append({
                "timestamp": timestamp,
                "ofi": ofi,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "mid_price": mid_price
            })
        
        return pd.DataFrame(ofi_data)
    
    def normalize_prices(
        self, 
        trades_df: pd.DataFrame, 
        method: str = "zscore"
    ) -> pd.DataFrame:
        """
        this methodd ormalises the price.

        """
        df = trades_df.copy()
        #using standardisdation zcore method to nomr 
        if method == "zscore":
            mean_price = df["price"].mean()
            std_price = df["price"].std()
            df["price_normalized"] = (df["price"] - mean_price) / std_price
        #using nirmalisation min-max method to norm     
        elif method == "minmax":
            min_price = df["price"].min()
            max_price = df["price"].max()
            df["price_normalized"] = (df["price"] - min_price) / (max_price - min_price)
        # using log returns to normaluse 
        elif method == "returns":
            df["price_normalized"] = np.log(df["price"] / df["price"].shift(1))
            df = df.dropna()
        
        return df
    
    def split_train_test(
        self, 
        trades_df: pd.DataFrame, 
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        train/test split using unanchored approach.
        """
        n = len(trades_df)
        split_idx = int(n * train_ratio)
        
        train_df = trades_df.iloc[:split_idx].copy()
        test_df = trades_df.iloc[split_idx:].copy()
        
        # restting the time_seconds for testing set
        if "time_seconds" in test_df.columns:
            test_df["time_seconds"] = (
                test_df["time"] - test_df["time"].iloc[0]
            ).dt.total_seconds()
        
        return train_df, test_df
    
    def aggregate_trades(
        self, 
        trades_df: pd.DataFrame, 
        method: str = "time",
        interval: float = 1.0
    ) -> pd.DataFrame:
        """
        summing up  trades into trade bars.

        """
        df = trades_df.copy() #coping to prev altering original data 
        
        if method == "time": # bats based upon time intervals
            df = df.set_index("time")
            bars = df.resample(f"{int(interval)}S").agg({
                "price": ["first", "max", "min", "last"],
                "size": "sum"
            })
            bars.columns = ["open", "high", "low", "close", "volume"]
            return bars.reset_index() #reseting index
        
        elif method == "volume": # bars based upon vol intervals
            df["cumulative_volume"] = df["size"].cumsum()
            df["bar_id"] = (df["cumulative_volume"] // interval).astype(int)
            
            #hroupby bar_id to create tbars
            bars = df.groupby("bar_id").agg({
                "time": "last",
                "price": ["first", "max", "min", "last"],
                "size": "sum"
            })
            bars.columns = ["time", "open", "high", "low", "close", "volume"]
            return bars.reset_index(drop=True) #resetting idx
        
        elif method == "tick": # bars based upon tick intrvals 
            df["bar_id"] = df.index // int(interval) # dividing idx by interval => get bar id as a res 
            # groupby bar_id to creattbars
            bars = df.groupby("bar_id").agg({
                "time": "last",
                "price": ["first", "max", "min", "last"],
                "size": "sum"
            })
            bars.columns = ["time", "open", "high", "low", "close", "volume"]
            return bars.reset_index(drop=True) #resetting idx
        
        return df
    
    def get_summary_statistics(self, 
                trades_df: pd.DataFrame) -> Dict:
        """
        ompute summary stats for trades data.
    
        """
        df = trades_df.copy()
        #if we aint fot classifed side => classify 
        if "classified_side" not in df.columns:
            df = self.classify_trades(df)
        
        # computing a a whole bunch of stats 
        stats = { #store the stats in a hashmap 
            "total_trades": len(df),
            "duration_seconds": df["time_seconds"].max() if "time_seconds" in df.columns else 0,
            # these are buy/sell counts and volumes and a lot more 
            "buy_trades": (df["classified_side"] == "buy").sum(),
            "sell_trades": (df["classified_side"] == "sell").sum(),
            "total_volume": df["size"].sum(),
            #avgs, median and ranges + vol 
            "avg_trade_size": df["size"].mean(),
            #meidan
            "median_trade_size": df["size"].median(),
            "price_range": (df["price"].min(), df["price"].max()),
            "avg_price": df["price"].mean(), #avg price calcu 
            "price_volatility": df["price"].std(),
        }
        
        # event interarrival times calculated below 
        if "time_seconds" in df.columns:
            # differencinf consec times => get int arriv times
            inter_arrival = df["time_seconds"].diff().dropna()
            stats["avg_inter_arrival"] = inter_arrival.mean()
            stats["std_inter_arrival"] = inter_arrival.std()
        
        return stats


if __name__ == "__main__":
    # if this is the entry poitnt => we have theexample usage 
    from data_retrieval import CoinbaseDataRetriever
    
    retriever = CoinbaseDataRetriever("BTC-USD")
    trades = retriever.get_trades(limit=1000)
    #preprocess and computing stats 
    processor = DataProcessor()
    processed_trades = processor.load_and_process_trades(trades)

    stats = processor.get_summary_statistics(processed_trades)
    # extracting evensts for Hawkes 
    event_times, event_types = processor.extract_hawkes_events(processed_trades)
    
