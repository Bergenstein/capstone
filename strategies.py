"""
this module includes the code logic for the backtesting engine for testing different strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    """this is the container that contains storage for all the backtest results that we get back from the engine."""
    strategy_name: str
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    fill_rate: float
    final_inventory: float
    num_trades: int
    avg_spread: float
    pnl_history: List[float] = field(default_factory=list)
    inventory_history: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    quote_history: List[Dict] = field(default_factory=list)


class BacktestEngine:
    """
    the actual backtesting engine that sims MM strats 
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0, #init cach 
        initial_inventory: float = 0.0,
        tick_size: float = 0.01,
        transaction_cost: float = 0.0001
    ):
        """
        init backtest engiine with some init params and init conds.
        """
        self.initial_cash = initial_cash #cache is 100k but can be latered 
        self.initial_inventory = initial_inventory #init inv to 0 
        self.tick_size = tick_size
        self.transaction_cost = transaction_cost
        
    def run_backtest(
        self,
        strategy: Any,
        market_data: pd.DataFrame,
        strategy_name: str = "Strategy"
    ) -> BacktestResult:
        """
        running the backtetsing engine for a specific stratgy based upon market daya 
        """
        # init state; starting with $$ and no inv at this point 
        cash = self.initial_cash
        inventory = self.initial_inventory
        pnl = 0.0
        # pnl history evolution 
        pnl_history = []
        inventory_history = [] #inv historty evolution 
        time_history = [] #tine evolution 
        trade_history = []
        quote_history = []
        
        total_fills = 0
        total_quotes = 0
        spread_sum = 0.0
        
        # sort  data by time => process events using correct order
        market_data = market_data.sort_values("time_seconds" if "time_seconds" in market_data.columns else "time")
        
        for idx, row in market_data.iterrows():
            if "time_seconds" in market_data.columns:
                t = row["time_seconds"]
            else:
                # calctime in seconds right from start if not avail
                t = (row["time"] - market_data.iloc[0]["time"]).total_seconds()
            
            mid_price = row["price"]
            market_side = row.get("classified_side", row.get("side", "buy"))
            market_size = row.get("size", 0.1)
            
            # retrieving our quotes based off strats using a try cache block
            try:
                bid, ask = strategy.get_quotes(t, mid_price, inventory)
            except Exception as e:
                continue
            
            # rounding so that the bid and asks are made based on the tick size and aion't tpoo much 
            bid = np.round(bid / self.tick_size) * self.tick_size
            ask = np.round(ask / self.tick_size) * self.tick_size
            spread = ask - bid
            
            #increment total quates and spread sum 
            total_quotes+= 1
            spread_sum+= spread
            
            # logging area here 
            quote_history.append({
                "time": t,
                "bid": bid,
                "ask": ask,
                "mid": mid_price,
                "inventory": inventory
            })
            
            # PROBABILISTIC FILL LOGIC:
            # We don't have real orderbook => prob model so that we simulate the logoc that happens in real markets as much as possible 
            filled = False
            fill_price = None
            fill_side = None
            
            #  spread as a percentage of the calculated mid price
            spread_pct = spread / mid_price
    
            base_fill_prob = np.exp(-spread_pct * 1000)
            
            if market_side == "sell":
                # A sell order came in and it mat indeed hit our bid if we are competitive => 
                # Check if our bid is close enough to mid . this is within 20bps

                if (mid_price-bid) / mid_price < 0.002:
                    # a random fill based on a specifed base rate prob 
                    if np.random.random() < base_fill_prob:
                        fill_price = bid
                        fill_side = "buy"
                        cash -= fill_price * market_size * (1 + self.transaction_cost)
                        inventory += market_size
                        filled = True
                        total_fills += 1
                
            elif market_side == "buy":
                # A buy order came in and might hit our ask if we are competitive
                # Check if our ask is close enough to mid (within 20bps)
                if (ask - mid_price) / mid_price < 0.002:
                    if np.random.random() < base_fill_prob:
                        fill_price = ask
                        fill_side = "sell"
                        cash += fill_price * market_size * (1 - self.transaction_cost)
                        inventory -= market_size
                        filled = True
                        total_fills += 1
            
            if filled:
                # Log the trade that happend here 
                trade_history.append({
                    "time": t,
                    "side": fill_side,
                    "price": fill_price,
                    "size": market_size,
                    "inventory_after": inventory
                })
                
                # Tell strategy abt this fill so it can update its state like times for hawkes calcs
                if hasattr(strategy, "record_event"):
                    try:
                        strategy.record_event(t, market_side)
                    except:
                        pass
            
            # Mark to market - update PnL based on curretn prices in the mkt
            portfolio_value = cash + inventory * mid_price
            pnl = portfolio_value - self.initial_cash
            
            pnl_history.append(pnl)
            inventory_history.append(inventory)
            time_history.append(t)
        
        # calcs final stats from backtest 
        total_pnl = pnl_history[-1] if pnl_history else 0.0
        
        # sharoe ratio computation !annualized => not to inflate metrics
        if len(pnl_history) > 1:
            returns = np.diff(pnl_history)
            sharpe = np.mean(returns)/ (np.std(returns) +1e-6) # avoids /0
        else:
            sharpe = 0.0 # if not enoygh data => 0 is our sharpe 
        
        # calc max drawdown from the pnl evolution 
        if len(pnl_history) > 0:
            cummax = np.maximum.accumulate(pnl_history)
            drawdowns = cummax - pnl_history
            max_drawdown = np.max(drawdowns)
        else:
            max_drawdown = 0.0
        
        # ffill rate that checks how many of our sent quates were actually filled 
        fill_rate = total_fills / total_quotes if total_quotes > 0 else 0.0
        
        # avg spread that we quoted all the way throughout the btest 
        avg_spread = spread_sum / total_quotes if total_quotes > 0 else 0.0
        # the entire backtest results obj is returned here 
        return BacktestResult(
            strategy_name=strategy_name,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            fill_rate=fill_rate,
            final_inventory=inventory,
            num_trades=total_fills,
            avg_spread=avg_spread,
            pnl_history=pnl_history,
            inventory_history=inventory_history,
            time_history=time_history,
            trade_history=trade_history,
            quote_history=quote_history
        )
    
    def compare_strategies(
        self,
        strategies: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        comparing various diff strategies on the exactb same market data and then returning a pandas df with metrics that are needed for each strat.
        """
        results = []
        
        #iterate through strats and then and run backtest for each one of them 
        for name, strategy in strategies.items():
            result = self.run_backtest(strategy, market_data, name)
            # append these all  to the array of res 
            results.append({
                "Strategy": name,
                "Total PnL": result.total_pnl, # pnl 
                "Sharpe Ratio": result.sharpe_ratio,
                "Max Drawdown": result.max_drawdown, #drawdown max ratio 
                "Fill Rate": result.fill_rate,
                "Num Trades": result.num_trades,
                "Final Inventory": result.final_inventory,
                "Avg Spread": result.avg_spread
            })
        
        return pd.DataFrame(results) #retur as a pandas df 
    
    def walk_forward_analysis(
        self,
        strategy_class: Any,
        calibrator: Any,
        market_data: pd.DataFrame,
        train_window: int = 500,
        test_window: int = 100,
        step_size: int = 100
    ) -> List[BacktestResult]:
        """
        walk forward nonanchrored backtest 
        
        """
        # res arr 
        results = []
        n = len(market_data) # length of market data 
        # sliding window logic here so that we do nonanchored walk forward properly 
        for start_idx in range(0, n - train_window - test_window, step_size):
            train_end = start_idx + train_window
            test_end = train_end + test_window
            # the break codition is met when test reach the end
            if test_end > n: break #exit loop 
            
            # retrieve and extarct training data from this sliding window
            train_data = market_data.iloc[start_idx:train_end]
            
            # calibrataion done on the on the training daya 
            event_times = train_data["time_seconds"].values
            event_types = (train_data["classified_side"] == "buy").astype(int).values
            
            # fit specific params on training data 
            params = calibrator.calibrate_exponential_kernel(event_times, event_types)
            
            # extracting test data from this wiodow of walkforward test 
            test_data = market_data.iloc[train_end:test_end]
            
            # creating a set of new strat instances that are tuned up with the params
            strategy = strategy_class(params)
            
            # bbacktesting done on test data 
            result = self.run_backtest(strategy, test_data, f"Window_{start_idx}")
            results.append(result)
        
        return results


