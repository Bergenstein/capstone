"""
Data Retrieval Module

Retrieves L2 orderbook data from Coinbase API.
"""

import pandas as pd
import time, json, os, requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple



class CoinbaseDataRetriever:
    """
    Retrieves real-time and historical L2 orderbook data from Coinbase.
    """

    def __init__(self, product_id: str = "BTC-USD"):
        """
        this method intialises the data retriever.
        
        """
        self.product_id = product_id #BTC-USD 
        #coinbase api 
        self.base_url = "https://api.exchange.coinbase.com"
        self.session = requests.Session() #making a session request to reuse the connectton if need be 
        
    def get_orderbook(self, level: int = 2) -> Dict:
        """
        retrieving L2 orderbook data/snapshot.
        """
        #the end point for orderbook data 
        endpoint = f"{self.base_url}/products/{self.product_id}/book"
        params = {"level": level} #level 2 to get capture the deeper level of data 
        #try catch block to handle if there is an exception with connection request
        try:
            res = self.session.get(endpoint, params=params) #getting res 
            res.raise_for_status() #get error if bad status 
            return res.json() # retrieve the json data for analysis 
        except requests.exceptions.RequestException as e:
            return None
    
    def get_trades(self, limit: int = 1000) -> pd.DataFrame:
        """
        Get most recent recent trades data.

        """
        endpoint = f"{self.base_url}/products/{self.product_id}/trades"
        
        all_trades = [] # an array to store all trade datya 
        before = None
        
        while len(all_trades) < limit:
            # this is to ensure that we get the max allowed num trades 
            # coinvase allows max 1000 upon a request
            params = {"limit": min(1000, limit - len(all_trades))}
            if before: # if before is set => then it is stored inside params 
                params["before"] = before
                
            try:
                #try catch block to check if any issues with connec 
                response = self.session.get(endpoint, params=params)
                response.raise_for_status()
                trades = response.json()
                
                if not trades: break #breaks if there is no trade. Do NOT continye the while loop
                    
                all_trades.extend(trades)
                before = trades[-1]["trade_id"]
                
                # this handles API rates limiting to make sure we don't get locked out 
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                break
        
        # the retrieved data is concverted to df from json for better analusis with pandas 
        df = pd.DataFrame(all_trades)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]) #valyes are converted properly and 
            df["price"] = df["price"].astype(float)
            df["size"] = df["size"].astype(float)
            df = df.sort_values("time").reset_index(drop=True) # soted by time 
            
        return df
    
    def get_historical_candles(
        self, 
        start: datetime, 
        end: datetime, 
        granularity: int = 60
    ) -> pd.DataFrame:
        """
        the particlar method retrieves historical price data.
    
        """
        endpoint = f"{self.base_url}/products/{self.product_id}/candles"
        
        # handling coinbase limits of 300/request
        max_candles = 300
        interval_seconds = granularity * max_candles # calcing intervals base on garnularity of trades here 
        
        all_candles = [] # storage for all candle data 
        current_start = start
        # looping via time intervals 
        while current_start < end:
            current_end = min(current_start + timedelta(seconds=interval_seconds), end)
            
            params = {
                "start": current_start.isoformat(),
                "end": current_end.isoformat(),
                "granularity": granularity
            }
            # try catch block to check if any issues with connec
            try:
                response = self.session.get(endpoint, params=params)
                response.raise_for_status()
                candles = response.json()
                # if we get historical candle data => we store inside the array 
                if candles:
                    all_candles.extend(candles)
                
                current_start = current_end
                time.sleep(0.5)  # dealing with API rate lmits again 
                
            except requests.exceptions.RequestException as e:
             
                break # break outta while loop when there is an issue 
        
        # converying to pandas df fr ease of processing and analysis 
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=["time", "low", "high", "open", "close", "volume"]
            )
            # seconds granularity 
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True) # sort by time and reset the idx 
            return df
        else:
            return pd.DataFrame()
    
    def stream_orderbook_changes(
        self, 
        duration_seconds: int = 3600,
        callback=None
    ) -> List[Dict]:
        """
        Stream L2 orderbook updates for a specific duration of time here it is set to 1 hoyr but can change 
        """
        snapshots = [] # to store all snapshots fro L2 orderbook data
        start_time = time.time() # start time here 
        # loop until the specific duration is indeed reached 
        while time.time() - start_time < duration_seconds:
            orderbook = self.get_orderbook(level=2) # level L2 data from OB Coinbase 
            
            if orderbook: #if we get the valid data 
                # store the snapshot of data with their specific timestamps inside the snapshot array for further analysus 
                snapshot = {
                    "timestamp": datetime.now(),
                    "bids": orderbook.get("bids", []),
                    "asks": orderbook.get("asks", []),
                    "sequence": orderbook.get("sequence", None)
                }
                
                snapshots.append(snapshot) #array storage 
                
                if callback: # if a callback morphism => it is called here 
                    callback(snapshot) #with snapshot as an arg 
            
            time.sleep(1)  # sample every single second
        
        return snapshots
    
    def extract_trade_events(self, trades_df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """
        extracting trade events from trades df we hav already processed.
        """
        if trades_df.empty: #if all empty =>  then return empty arrs here
            return [], []
        
        # we are here convertung time to secs all the way from the beginning( since start!!)
        start_time = trades_df["time"].iloc[0]
        #rimestamps 
        timestamps = (trades_df["time"] - start_time).dt.total_seconds().tolist()
        sides = trades_df["side"].tolist() #list of all trades (bid/ask)
        
        return timestamps, sides #tuple returned
    
    def get_product_info(self) -> Dict:
        """
        thsi method retrieved information such as size of tick etc.

        """
        #defining the endpoint so that we can retreive the info via making reqs
        endpoint = f"{self.base_url}/products/{self.product_id}"
        # try catch block to handle  issues with connc  reqs
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
         
            return None


class DataCache:
    """
    TYhis class deals with and manages caching the data retrieved straight to local storage.
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        """
        initialisers for cached data .

        """
        self.cache_dir = cache_dir #the local dir to store it 
        os.makedirs(cache_dir, exist_ok=True) #if the dir doesnt exits => we make it
    
    def save_trades(self, trades_df: pd.DataFrame, filename: str):
        """trades are saved in csv format."""
        filepath = f"{self.cache_dir}/{filename}" #path t file 
        trades_df.to_csv(filepath, index=False)

    
    def load_trades(self, filename: str) -> Optional[pd.DataFrame]:
        """loading data from cache instead of retrieving it again."""
        filepath = f"{self.cache_dir}/{filename}"
        try:
            df = pd.read_csv(filepath) #the path  to our file 
            df["time"] = pd.to_datetime(df["time"]) #if time col exists => convert to df(pandas)
            return df
        except FileNotFoundError:
            print(f" we were unable for find cache file {filepath} not found")
            return None
    
    def save_orderbook_snapshots(self, snapshots: List[Dict], filename: str):
        """svae orderbook L2 snapshots  => json."""
        filepath = f"{self.cache_dir}/{filename}" #name of the path 
        #utilsiing the open module to write json data 
        with open(filepath, "w") as f:
            # convrts df objects to str 
            snapshots_serializable = []
            for snap in snapshots:
                snap_copy = snap.copy()
                snap_copy["timestamp"] = snap_copy["timestamp"].isoformat()
                snapshots_serializable.append(snap_copy)
            json.dump(snapshots_serializable, f)
     
    
    def load_orderbook_snapshots(self, filename: str) -> Optional[List[Dict]]:
        """load L2 OB snapshots from json."""
        filepath = f"{self.cache_dir}/{filename}"
        try:
            with open(filepath, "r") as f:
                snapshots = json.load(f)
                # covert timestamp strings back to datetime
                for snap in snapshots:
                    snap["timestamp"] = datetime.fromisoformat(snap["timestamp"])
                return snapshots
        except FileNotFoundError:
            print(f" we were unable for find cache file {filepath} not found")
            return None

