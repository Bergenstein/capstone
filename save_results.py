"""
saving res and generating all reports for the models all 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# importing the modules 
from data_retrieval import CoinbaseDataRetriever, DataCache
from data_processing import DataProcessor
from univariate_hawkes import UnivariateHawkesCalibrator, UnivariateHawkesMarketMaker
from multivariate_hawkes import MultivariateHawkesCalibrator, HawkesMarketMaker
from avellaneda_stoikov import AvellanedaStoikovMM
from strategies import BacktestEngine
from metrics import PerformanceMetrics
from visualization import Visualizer


def main():
    # this is the results dir where we store data 
    os.makedirs("results", exist_ok=True)
    
    print("=" * 80)
    print("SAVING RESULTS AND GENERATING REPORT")
    print("=" * 80)
    print()
    
    # Load cached data
    cache = DataCache()
    trades_df = cache.load_trades("BTC_USD_trades_5000.csv")
    
    if trades_df is None:
        print("There is no cached data yet => call 'run_tests.py' first please.")
        return
    
    # processing the data
    processor = DataProcessor()
    processed = processor.load_and_process_trades(trades_df)
    processed = processor.classify_trades(processed)
    train_df, test_df = processor.split_train_test(processed, 0.7)
    
    # calibrrating the models all based on hawkes events 
    train_times, train_types = processor.extract_hawkes_events(train_df)
    uni_calibrator = UnivariateHawkesCalibrator()
    uni_params = uni_calibrator.calibrate(train_times)
    
    exp_calibrator = MultivariateHawkesCalibrator()
    exp_params = exp_calibrator.calibrate_exponential_kernel(train_times, train_types)
    
    pl_calibrator = MultivariateHawkesCalibrator()
    pl_params = pl_calibrator.calibrate_powerlaw_kernel(train_times, train_types)
    
    # saving calibrated data and params to csv format 
    calib_df = pd.DataFrame([
        {"Model": "Univariate", "Log-Likelihood": uni_calibrator.log_likelihood, **uni_params},
        {"Model": "Exponential", "Log-Likelihood": exp_calibrator.log_likelihood, **exp_params},
        {"Model": "Power-law", "Log-Likelihood": pl_calibrator.log_likelihood, **pl_params}
    ])
    calib_df.to_csv("results/calibration_parameters.csv", index=False)
    
    # performing and creating strategies all 
    strategies = {
        "Univariate Hawkes": UnivariateHawkesMarketMaker(uni_params, risk_aversion=0.1),
        "Hawkes Exponential": HawkesMarketMaker(exp_params, "exponential", risk_aversion=0.1, imbalance_sensitivity=0.5),
        "Hawkes Power-law": HawkesMarketMaker(pl_params, "powerlaw", risk_aversion=0.1, imbalance_sensitivity=0.5),
        "Avellaneda-Stoikov": AvellanedaStoikovMM(risk_aversion=0.1, volatility=0.0002, time_horizon=test_df["time_seconds"].max())
    }
    
    # running the backtesting engine based off different strategies (all 4 up)
    engine = BacktestEngine()
    results = {}
    for name, strategy in strategies.items():
        results[name] = engine.run_backtest(strategy, test_df, name)
    
    # saving the performance metrics and the comparison between different models and the strategies run to csv format locally 
    comparison_df = engine.compare_strategies(strategies, test_df)
    comparison_df.to_csv("results/performance_comparison.csv", index=False)
    
    # checking each strategy and saving the in depth report for each 
    for name, result in results.items():
        safe_name = name.replace(" ", "_").replace("-", "_").lower()
        
        # pnl evolution histroy is saved here 
        pnl_df = pd.DataFrame({
            "time": result.time_history,
            "pnl": result.pnl_history,
            "inventory": result.inventory_history
        })
        # save it to csv after retrieving it
        pnl_df.to_csv(f"results/{safe_name}_pnl.csv", index=False)
        
        # the trading history is also retreived and saved here if any 
        if result.trade_history:
            trades_df = pd.DataFrame(result.trade_history)
            trades_df.to_csv(f"results/{safe_name}_trades.csv", index=False) #save to CSV here
    
    
    # visualusations and graphs

    viz = Visualizer()
    plt.ioff()
    
    # pnl comparison graphs using matplot lib library 
    viz.plot_pnl_comparison(results, figsize=(14, 6))
    plt.savefig("results/pnl_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # drawdown evolution comparison graphs using matplot lib library
    viz.plot_drawdown_comparison(results, figsize=(14, 6))
    plt.savefig("results/drawdown_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Inventory
    viz.plot_inventory_evolution(results, figsize=(14, 6))
    plt.savefig("results/inventory_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Returns distribution
    viz.plot_returns_distribution(results, figsize=(14, 6))
    plt.savefig("results/returns_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Q-Q plots
    viz.plot_qq_plots(results, figsize=(12, 10))
    plt.savefig("results/qq_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Performance metrics
    viz.plot_performance_metrics(results, figsize=(14, 10))
    plt.savefig("results/performance_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # calibration comp 
    params_comparison = {
        "Univariate": uni_params,
        "Exponential": exp_params,
        "Power-law": pl_params
    }
    viz.plot_calibration_comparison(params_comparison, figsize=(14, 6))
    plt.savefig("results/calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
