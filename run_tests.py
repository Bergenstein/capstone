"""
this is Main file for running tests. It Runs tests on Coinbase data to compafe Hawkes processes with AS MM based strategies 
"""
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from data_retrieval import CoinbaseDataRetriever, DataCache
from data_processing import DataProcessor
from univariate_hawkes import UnivariateHawkesCalibrator, UnivariateHawkesMarketMaker
from multivariate_hawkes import MultivariateHawkesCalibrator, HawkesMarketMaker
from avellaneda_stoikov import AvellanedaStoikovMM
from calibration import EMCalibrator
from strategies import BacktestEngine
from metrics import PerformanceMetrics
from visualization import Visualizer
import traceback

def main():
    """
    Main entry for running the tests 
    """

    # Configs 
    ticker = "BTC-USD"
    trades_num = 5000  
    activate_cache = True # so that we dont have to dowbnload data again and again
    
    # Init 
    data_retr = CoinbaseDataRetriever(ticker) #data retriever obj 
    processor = DataProcessor() # data processor obj to create an orderbook obj
    cache = DataCache() #uses cache to save data 
    viz = Visualizer() #visaulisaton obj 
    engine = BacktestEngine(initial_cash=100000.0, transaction_cost=0.0001)
   
    # retreiving data 1st
    cache_filename = f"{ticker.replace('-', '_')}_trades_{trades_num}.csv"
    
    if activate_cache: # we shall load cache if we can 
        df_to_trad = cache.load_trades(cache_filename)
    else: df_to_trad = None
    
    if df_to_trad is None:
        # retrieving data from API
        df_to_trad = data_retr.get_trades(limit=trades_num)
        
        if df_to_trad is None or len(df_to_trad) == 0: return
        
        if activate_cache:
            cache.save_trades(df_to_trad, cache_filename)
    else:
        print(f"successfully loaded this much {len(df_to_trad)} trades from the cache")

    # process data 
    
    # Procesing the tradess 
    processed_trades = processor.load_and_process_trades(df_to_trad)
    processed_trades = processor.classify_trades(processed_trades, method="side")
    
    # retrieve and exract Hawkes data events
    event_times, event_types = processor.extract_hawkes_events(processed_trades)
    
    # get sum stats statistics
    stats = processor.get_summary_statistics(processed_trades)
    
    # Split train/test (using 70% for training and now using a rolling window)
    train_df, test_df = processor.split_train_test(processed_trades, train_ratio=0.7)

    #calicbraring the mdoel here 
    train_times, train_types = processor.extract_hawkes_events(train_df)
    T_train = train_times[-1]
    
    # Testing now on Univariate Hawkes model

    uni_calibrator = UnivariateHawkesCalibrator()
    uni_params = uni_calibrator.calibrate(train_times)
    
    # Testing is now done on Multivariate Hawkes with Exp decay kernel

    exp_calibrator = MultivariateHawkesCalibrator()
    exp_params = exp_calibrator.calibrate_exponential_kernel(train_times, train_types)

    
    # multivariate Hawkes testing with (Power-law) as kernel 

    pl_calibrator = MultivariateHawkesCalibrator()
    pl_params = pl_calibrator.calibrate_powerlaw_kernel(train_times, train_types)
    
    # Avellaneda-Stoikov (AS) (with params estimates from data)
   
    avg_arrival_rate = len(train_times) / T_train
    avg_price = train_df["price"].mean()
    volatility = train_df["price"].pct_change().std()

    
    # strategies section 

    strategies = {}
    
    # Univariate Hawkes
    strategies["Univariate Hawkes"] = UnivariateHawkesMarketMaker(
        params=uni_params,
        risk_aversion=0.01,  # tigher spreads here due to lowerr risk aversion baked in
        inventory_target=0.0
    )
    
    # Multivariate Hawkes with (Exponential decay) kernel
    strategies["Hawkes Exponential"] = HawkesMarketMaker(
        params=exp_params,
        kernel_type="exponential",
        risk_aversion=0.01,  # Lower risk aversion here as well so  => tigher spreads 
        inventory_target=0.0,
        imbalance_sensitivity=1.0  # more sensitive to order flow imbalance 
    )
    
    # Multivariate Hawkes with (Power-law) kernel
    strategies["Hawkes Power-law"] = HawkesMarketMaker(
        params=pl_params,
        kernel_type="powerlaw",
        risk_aversion=0.01,  # Lower risk aversion here as well so  => tigher spreads
        inventory_target=0.0,
        imbalance_sensitivity=1.0  # Higher sensitivity to OFI (order flow imbalance microstructure)
    )
    
    # Avellaneda-Stoikov (AS model)
    strategies["Avellaneda-Stoikov"] = AvellanedaStoikovMM(
        risk_aversion=0.01,  # Lower risk aversion
        order_arrival_rate=avg_arrival_rate,
        order_fill_probability=2.0,  # Higher probability of filling order 
        volatility=max(volatility, 0.0002),  # num stabilkity 
        time_horizon=test_df["time_seconds"].max()
    )

    
    # =============================================== Backtesting mechanism here 
    # ==========================================================
    
    results = {}
    for name, strategy in strategies.items():
        result = engine.run_backtest(strategy, test_df, name)
        results[name] = result
    
    # Res 
    # ======================================================
   
    comparison_df = engine.compare_strategies(
        {name: strategy for name, strategy in strategies.items()},
        test_df
    )
    
    
    # Best strategy based on Pnl metric and sharpe metric 
    best_strategy = comparison_df.loc[comparison_df["Total PnL"].idxmax(), "Strategy"]

    best_sharpe = comparison_df.loc[comparison_df["Sharpe Ratio"].idxmax(), "Strategy"]
    
    # Compute OFI to be used with Hawkes based strats engine 
    test_times, test_types = processor.extract_hawkes_events(test_df)
    
    # Some sampling of the  OFI  microstructure at regular intervals
    sample_times = np.linspace(test_times[0], test_times[-1], 100)
    
    exp_strategy = strategies["Hawkes Exponential"]
    pl_strategy = strategies["Hawkes Power-law"]
    
    ofi_exp = []
    ofi_pl = []
    
    for t in sample_times:
        # record events up to the timestep of t
        for i, event_time in enumerate(test_times):
            if event_time > t:
                break
            event_type = "buy" if test_types[i] == 1 else "sell" #logic on when to enter buy or short 
            exp_strategy.record_event(event_time, event_type)
            pl_strategy.record_event(event_time, event_type)
        
        ofi_exp.append(exp_strategy.order_flow_imbalance(t)) # we are gettig the ofi values here 
        ofi_pl.append(pl_strategy.order_flow_imbalance(t))
    
    ofi_exp = np.array(ofi_exp)
    ofi_pl = np.array(ofi_pl)
    
    # Compute price dynamics evolution 
    price_samples = np.interp(sample_times, test_df["time_seconds"].values, test_df["price"].values)
    price_changes = np.diff(price_samples)
    
    # OFI accuracy
    if len(ofi_exp) > 1:
        ofi_exp_trimmed = ofi_exp[:-1]
        ofi_pl_trimmed = ofi_pl[:-1]
        
        acc_exp = PerformanceMetrics.order_flow_imbalance_accuracy(ofi_exp_trimmed, price_changes)
        acc_pl = PerformanceMetrics.order_flow_imbalance_accuracy(ofi_pl_trimmed, price_changes)
        
    # visualise metrics here 
    
    # visualisation to perform profit and loss (PnL comps
    viz.plot_pnl_comparison(results)
    
    # Drawdown metric comprisson 
    viz.plot_drawdown_comparison(results)
    
    # Inventory evolution comparison 
    viz.plot_inventory_evolution(results)
    
    # calcing Returns distr and plotting it here 
    viz.plot_returns_distribution(results)
    
    # Q-Q plots
    viz.plot_qq_plots(results)
    
    # Performance metrics calculations and plottng 
    viz.plot_performance_metrics(results)
    
    # OFI (order flow imbalance) visuals ation!!
    if len(ofi_exp) > 0:
        viz.plot_order_flow_imbalance(sample_times, ofi_exp, price_samples)
    
    # comparing model Calibration comparison
    params_comparison = {
        "Univariate": uni_params,
        "Exponential": exp_params,
        "Power-law": pl_params
    }
    viz.plot_calibration_comparison(params_comparison)


    
    for name, result in results.items():
        if len(result.pnl_history) < 3:
            continue
        
        returns = np.diff(result.pnl_history)
        normality = PerformanceMetrics.normality_test(returns)




if __name__ == "__main__": # entry point for the main program 
    try:
        main()
    except KeyboardInterrupt:
        print("test has been interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f" the error faced is: {e}")
        traceback.print_exc()
        sys.exit(1)
