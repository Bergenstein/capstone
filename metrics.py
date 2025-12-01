"""
This module calculates various financial performance metrics to check models performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats


class PerformanceMetrics:
    """
    Calculuates the performance metrics for diff trading strategies.
    """
    @staticmethod
    def sharpe_r(
        rets: np.ndarray,
        rf: float = 0.0,
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        the method simply calculates the  Sharpe ratio.
        """
        if len(rets) < 2: #npt possible to compute sharpe on this 
            return 0.0
        # excess retirn over risk free rate 
        excess_rets = rets - rf
        #mean of strategy retiurns 
        mean_return = np.mean(excess_rets)
        # std of strategy returns 
        std_return = np.std(excess_rets)
        
        if std_return == 0:
            return 0.0
        #calcs sharpe ratio here
        sharpe = mean_return / std_return
        # annualisation which isn't done in the org code
        if periods_per_year:
            sharpe *= np.sqrt(periods_per_year)
        
        return sharpe
    
    @staticmethod
    def sortino_ratio(
        rets: np.ndarray,
        rf: float = 0.0,
        target_return: float = 0.0
    ) -> float:
        """
        Sortio ratio calculation 
        """
        if len(rets) < 2: #npt possible to compute sortino on this
            return 0.0
        
        excess_rets = rets - rf # excess returns over risk-free rate
        mean_return = np.mean(excess_rets)
        
        # calculatig downside deviation 
        downside_rets = excess_rets[excess_rets < target_return]
        if len(downside_rets) == 0:
            return np.inf # we cannot compute => results in inf div by zero /0
        # cakculate the std of downside returns (negative)
        downside_std = np.sqrt(np.mean((downside_rets - target_return) ** 2))
        
        if downside_std == 0:
            return 0.0
        # calkculate sortino 
        return mean_return / downside_std
    
    @staticmethod
    def max_drawdown(pnl_series: np.ndarray) -> Tuple[float, int, int]:
        """
        this method ompute maximum drawdown from the equity return drawdown evolution.
        """
        if len(pnl_series) == 0: # npt possible to compute drawdown
            return 0.0, 0, 0
        
        # maximum of the cumu ret to date fro pnl 
        cummax = np.maximum.accumulate(pnl_series)
        #diff between max and the timeseries pnl 
        drawdowns = cummax - pnl_series
        
        #max drawdown values, when it starts and stops 
        max_dd = np.max(drawdowns)
        end_idx = np.argmax(drawdowns) #the start of the drawdown 
        # and the end of the drawdown
        start_idx = np.argmax(pnl_series[:end_idx+1]) if end_idx > 0 else 0
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calmar_ratio(
        rets: np.ndarray,
        pnl_series: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calmar ratio (strattegy return / max_drawdown) calculated above.
        """
        if len(rets) == 0: # npt possible to compute calmar /0 
            return 0.0
        #annualse the returns (replace this dude because we shall not annualise based ona small sample)
        annualized_return = np.mean(rets) * periods_per_year
        #max_drawdown only from the tuple
        max_dd, _, _ = PerformanceMetrics.max_drawdown(pnl_series)
        
        if max_dd == 0:
            return np.inf
        #calculate calmar ratio 
        
        return annualized_return / max_dd
    
    @staticmethod
    def value_at_risk(
        rets: np.ndarray,
        confidence_level: float = 0.95 # 5% at risk (the extreme tails)
    ) -> float:
        """
        calc value at Risk (VaR) with 95% CI and 5% extreme tails.
        """
        if len(rets) == 0: # npt possible to calc VaR
            return 0.0
        #simple method to calc value at risk 
        return np.percentile(rets, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_value_at_risk(
        rets: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        calc conditional Value at Risk (CVaR). This is simply looking at the returns where  the returns are less than the value at risk. AKA exxpected Shortfall.
        """
        if len(rets) == 0: # npt possible to calc CVaR or Var 
            return 0.0
        #retrieving VAR
        var = PerformanceMetrics.value_at_risk(rets, confidence_level)
        cvar = np.mean(rets[rets <= var]) # expected shortfall calculattions 
        
        return cvar
    
    @staticmethod
    def hit_ratio(trades: List[Dict]) -> float:
        """
        calculates hit ratio. this is the same as proportion of profitable trades.
        """
        if len(trades) == 0: #impossible to get hit ratio if no trades 
            return 0.0
        
        #it is profitable if the ponl>0 so we just sum over them 
        profitable = sum(1 for t in trades if t.get("pnl", 0) > 0)
        # and divide by the aggre trades to get hit ratio 
        return profitable / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """
        compute profit factor (sum of profits from strategy / sum of negative retruns from strategy).
    
        """
        if len(trades) == 0: #no trades => impossible to calc profit factor
            return 0.0
        
        #summing over all + and - returns of the strategy to get profit and loss for the profit factor 
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        
        if gross_loss == 0: # this is literally that all trades are porfitable which is impossible so we spit out info 
            return np.inf if gross_profit > 0 else 0.0
        
        #profit factor calc 
        return gross_profit / gross_loss
    
    @staticmethod
    def inventory_turnover(
        inventory_history: np.ndarray,
        time_history: np.ndarray
    ) -> float:
        """
        this computes the  inventory turnover rate.
        """
        if len(inventory_history) < 2: #npt possible to compute turnover
            return 0.0
        # checks if the inventory has changed ovet rime at all or not 
        inventory_changes = np.abs(np.diff(inventory_history))
        time_diffs = np.diff(time_history)
        # the entire time history -1 - 0 gives us that 
        total_time = time_history[-1] - time_history[0]
        total_turnover = np.sum(inventory_changes)
        
        if total_time == 0: # to avoid div by zero /0
            return 0.0
        
        return total_turnover / total_time
    
    @staticmethod
    def order_flow_imbalance_accuracy(
        predicted_ofi: np.ndarray,
        actual_price_change: np.ndarray
    ) -> Dict[str, float]:
        """
        calculates the  accuracy of OFI microstructure predictions.
        """
        if len(predicted_ofi) != len(actual_price_change):
            raise ValueError("the arrays don't the the same size")
        
        if len(predicted_ofi) == 0: #there is no data so => we cannot cals anything 
            return {"accuracy": 0.0, "correlation": 0.0}
        
        # accuracy of direction 
        # prediction 1st 
        predicted_direction = np.sign(predicted_ofi)
        #actial dir 
        actual_direction = np.sign(actual_price_change)
        # accuracy that check if the two are equal and spits out a score (mean)
        accuracy = np.mean(predicted_direction == actual_direction)
        
        # calculating corr if the std>0 for both arrs 
        if np.std(predicted_ofi) > 0 and np.std(actual_price_change) > 0:
            correlation = np.corrcoef(predicted_ofi, actual_price_change)[0, 1]
        else: correlation = 0.0 #zero correl 
        #the dict that stores the two 
        return {
            "accuracy": accuracy,
            "correlation": correlation
        }
    
    @staticmethod
    def normality_test(rets: np.ndarray) -> Dict[str, float]:
        """
        tests ret for normality (to see if they follow a normal distr)
        """
        if len(rets) < 3:
            return {
                "shapiro_statistic": 0.0,
                "shapiro_pvalue": 1.0,
                "skewness": 0.0,
                "kurtosis": 0.0
            }
        
        # shapiro-Wilk test to check for normality 
        shapiro_stat, shapiro_p = stats.shapiro(rets)
        
        # skewness and kurtosis (excess kurtoisis) or skewness (left/right)
        skew = stats.skew(rets)
        kurt = stats.kurtosis(rets)
        
        return {
            "shapiro_statistic": shapiro_stat,
            "shapiro_pvalue": shapiro_p,
            "skewness": skew,
            "kurtosis": kurt
        }
    
    @staticmethod
    def comprehensive_report(
        pnl_history: List[float],
        inventory_history: List[float],
        time_history: List[float],
        trades: List[Dict],
        initial_capital: float = 100000.0 #starting networth 
    ) -> pd.DataFrame:
        """
        this part generate a performance report based off the strategies coded up here .
        """
        pnl_array = np.array(pnl_history)
        inventory_array = np.array(inventory_history)
        time_array = np.array(time_history)
        
        # here we are doing rets calcs 
        rets = np.diff(pnl_array) if len(pnl_array) > 1 else np.array([])
        
        # a hashmap used in order to store performance metrics calculated
        metrics = {}
        
        # pml metrics calculation 
        metrics["Total PnL"] = pnl_array[-1] if len(pnl_array) > 0 else 0.0
        metrics["Return %"] = (metrics["Total PnL"] / initial_capital) * 100 #mult by 100 to get perentage 
        
        # sharpe & sortino (risk adjusted strat rets ) calc here 
        metrics["Sharpe Ratio"] = PerformanceMetrics.sharpe_r(rets)
        metrics["Sortino Ratio"] = PerformanceMetrics.sortino_ratio(rets)
        
        # max drawdown calc here 
        max_dd, _, _ = PerformanceMetrics.max_drawdown(pnl_array)
        metrics["Max Drawdown"] = max_dd
        metrics["Max Drawdown %"] = (max_dd / initial_capital) * 100
        
        # value at risk and coditional VaR based on 95% CI (both)
        metrics["VaR 95%"] = PerformanceMetrics.value_at_risk(rets, 0.95)
        metrics["CVaR 95%"] = PerformanceMetrics.conditional_value_at_risk(rets, 0.95)
        
        # trading metrics are calculated here 
        metrics["Num Trades"] = len(trades)
        metrics["Inventory Turnover"] = PerformanceMetrics.inventory_turnover(
            inventory_array, time_array
        )
        
        # final state of te inventory 
        metrics["Final Inventory"] = inventory_array[-1] if len(inventory_array) > 0 else 0.0
        
        # tests to see if the returns are normal
        if len(rets) > 2:
            normality = PerformanceMetrics.normality_test(rets)
            metrics["Shapiro-Wilk p-value"] = normality["shapiro_pvalue"]
            metrics["Skewness"] = normality["skewness"]
            metrics["Kurtosis"] = normality["kurtosis"]
        
        # store in pandas df 
        df = pd.DataFrame([metrics]).T
        df.columns = ["Value"]
        
        return df
