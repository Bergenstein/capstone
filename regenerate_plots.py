"""
This script simply regenerates all visaluations from the CSV without needing to recompute the process
Can be run using: 
    python3 regenerate_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
from visualization import Visualizer

@dataclass
class BacktestResult:
    """results class for just visualisation """
    #all params needed so we do plotting 
    strategy_name: str
    total_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    fill_rate: float
    num_trades: int
    final_inventory: float
    avg_spread: float
    time_history: List[float]
    pnl_history: List[float]
    inventory_history: List[float]
    trade_history: List[dict]

def load_results_from_csv():
    """retrieve saved results from local csv files"""
    
    # performance comparison 
    comparison_df = pd.read_csv("results/performance_comparison.csv")
    
    # this maps to strategy names one by one 
    strategy_files = {
        "univariate_hawkes": "Univariate Hawkes",
        "hawkes_exponential": "Hawkes Exponential",
        "hawkes_power_law": "Hawkes Power-law",
        "avellaneda_stoikov": "Avellaneda-Stoikov"
    }
    #results hashmap to store the results for plotting 
    results = {}
    
    for file_prefix, display_name in strategy_files.items():
        # load PnL history evolution 
        pnl_df = pd.read_csv(f"results/{file_prefix}_pnl.csv")
        
        # load trade history evolution for local file via try catch block 
        try:
            trades_df = pd.read_csv(f"results/{file_prefix}_trades.csv")
            trade_history = trades_df.to_dict('records')
        except FileNotFoundError:
            trade_history = []
        
        # get/retrieve metrics from the comparison for plotting prep 
        metrics = comparison_df[comparison_df['Strategy'] == display_name].iloc[0]
        
        # create resulting obj 
        result = BacktestResult(
            strategy_name=display_name,
            total_pnl=metrics['Total PnL'],
            sharpe_ratio=metrics['Sharpe Ratio'],
            sortino_ratio=metrics.get('Sortino Ratio', 0.0),
            max_drawdown=metrics['Max Drawdown'],
            fill_rate=metrics['Fill Rate'],
            num_trades=int(metrics['Num Trades']),
            final_inventory=metrics['Final Inventory'],
            avg_spread=metrics['Avg Spread'],
            time_history=pnl_df['time'].tolist(),
            pnl_history=pnl_df['pnl'].tolist(),
            inventory_history=pnl_df['inventory'].tolist(),
            trade_history=trade_history
        )
        
        results[display_name] = result #this is used later for plotting 
    
    return results, comparison_df #return the tuples 

def load_calibration_params():
    """load calibration params"""
    calib_df = pd.read_csv("results/calibration_parameters.csv")
    # create a hashmap that is mapping model names to their precise params
    params_dict = {}
    for _, row in calib_df.iterrows(): #via itterrows 
        model_name = row['Model']
        params = row.drop(['Model', 'Log-Likelihood']).to_dict()
        params_dict[model_name] = params
    return params_dict

def regenerate_all_plots(results, params_dict):
    """regen all plots from the saved data if possible """
    
    vis_obj = Visualizer()
    plt.ioff()  # saving plots not displating them
    
    # 1. plotting pnl evolution comparison here!!
    vis_obj.plot_pnl_comparison(results, figsize=(14, 6))
    plt.savefig("results/pnl_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved pnl_comparison.png")
    
    # 2. plotting drawdown evolution comparison here!!!!
    vis_obj.plot_drawdown_comparison(results, figsize=(14, 6))
    plt.savefig("results/drawdown_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved drawdown_comparison.png")
    
    # plottig inventory evolution here!!
    vis_obj.plot_inventory_evolution(results, figsize=(14, 6)) #hyperparams
    plt.savefig("results/inventory_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved inventory_evolution.png")
    
    # plotting ret distributions here 
    vis_obj.plot_returns_distribution(results, figsize=(14, 6))
    plt.savefig("results/returns_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved returns_distribution.png")
    
    # QQ plots to plot 
    vis_obj.plot_qq_plots(results, figsize=(12, 10))
    plt.savefig("results/qq_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
   
    
    # performance metrics to plot 
    vis_obj.plot_performance_metrics(results, figsize=(14, 10))
    plt.savefig("results/performance_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    
    # 7. Calibration Comparison
    vis_obj.plot_calibration_comparison(params_dict, figsize=(14, 6))
    plt.savefig("results/calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    results, _ = load_results_from_csv()
    params_dict = load_calibration_params()
    
    # Regenerate plots
    regenerate_all_plots(results, params_dict)


if __name__ == "__main__":
    main()
