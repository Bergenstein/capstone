"""
Visualization Module

Plotting functions for market making analysis and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from scipy import stats


class Visualizer:
    """
    Visualization tools for market making strategies.
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        init
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def plot_pnl_comparison(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        pnl plotting 
        """
        #setting up figure and axis here 
        fig, ax = plt.subplots(figsize=figsize)
        # iterating ovet the results here via enum so that we get priper colours to use 
        for i, (name, result) in enumerate(results.items()):
            ax.plot(
                result.time_history,
                result.pnl_history,
                label=name,
                linewidth=2,
                color=self.colors[i % len(self.colors)]
            )
        
        # setting axis labels and title here
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Cumulative PnL ($)", fontsize=12)
        ax.set_title("Cumulative PnL Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_comparison(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        plotting the drawdown comparison here the we have calculated in other modules across diff modules and strategies 
        """
        fig, ax = plt.subplots(figsize=figsize)
        # enumerating over results here so that we retrieve the retirn via pnl and then use the return in a bid to calc drawdown
        for i, (name, result) in enumerate(results.items()):
            pnl = np.array(result.pnl_history)
            cummax = np.maximum.accumulate(pnl)
            drawdown = cummax - pnl #drawdown calc logic 
            #plotting the drawdown here
            ax.plot(
                result.time_history,
                drawdown,
                label=name,
                linewidth=2,
                color=self.colors[i % len(self.colors)]
            )
        #setting axis labels and title here
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Drawdown ($)", fontsize=12)
        ax.set_title("Drawdown Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_inventory_evolution(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        plotting inventory evolution over time here 
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # setting colours and labels; nothing special here 
        for i, (name, result) in enumerate(results.items()):
            ax.plot(
                result.time_history,
                result.inventory_history,
                label=name,
                linewidth=2,
                color=self.colors[i % len(self.colors)]
            )
        #setting axis and labels here as well with all simple values 
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Inventory", fontsize=12)
        ax.set_title("Inventory Evolution", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        #tight layout and showing 
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        plotting returns distribution in this method of the plotting class 
        """
        fig, axes = plt.subplots(1, len(results), figsize=figsize)
        
        if len(results) == 1:
            axes = [axes]
        
        # calculaying the returns 
        for i, (name, result) in enumerate(results.items()):
            returns = np.diff(result.pnl_history)
            # setting up histograms here for doing plotting (same all across the moduke)
            axes[i].hist(returns, bins=50, alpha=0.7, color=self.colors[i % len(self.colors)], edgecolor='black') #black colour. 
            axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2) #red colour maybe change it mate? experiement more 
            axes[i].set_xlabel("Returns", fontsize=10)
            axes[i].set_ylabel("Frequency", fontsize=10)
            axes[i].set_title(name, fontsize=12, fontweight="bold")
            axes[i].grid(True, alpha=0.3)
        #setting layout and showing the plot rightaway using .show()
        plt.tight_layout()
        plt.show()
    
    def plot_qq_plots(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        this method plot Q-Q plots 
        """
        # retrieving number of strategies here for 2 columns layout 
        n_strategies = len(results)
        n_cols = 2
        #num rows calc logic 
        n_rows = (n_strategies + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_strategies > 1 else [axes]
        #inumterating over results here so that we retrieve the retirn via pnl and then use it to calc QQ plot 
        for i, (name, result) in enumerate(results.items()):
            returns = np.diff(result.pnl_history)
            # QQ is plotted here using scipy stats package imported abobve
            stats.probplot(returns, dist="norm", plot=axes[i])
            axes[i].set_title(f"{name} Q-Q Plot", fontsize=12, fontweight="bold")
            axes[i].grid(True, alpha=0.3)
        
        # iterating ovee the rest of the axis and shuts them off if unused by mow 
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        plotting the bar charts of KPM
        """
        names = list(results.keys())
        
        # extracting metrics
        total_pnl = [r.total_pnl for r in results.values()]
        sharpe = [r.sharpe_ratio for r in results.values()]
        max_dd = [r.max_drawdown for r in results.values()]
        fill_rate = [r.fill_rate * 100 for r in results.values()]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # cumulative  PnL is plottyed here after getting retrieved 
        axes[0, 0].bar(names, total_pnl, color=self.colors[:len(names)])
        axes[0, 0].set_ylabel("Total PnL ($)", fontsize=10)
        axes[0, 0].set_title("Total PnL", fontsize=12, fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Sharpe Ratio is plottyed here after getting retrieved. It isnt annualised in a bid to not inflate pnl 
        axes[0, 1].bar(names, sharpe, color=self.colors[:len(names)])
        axes[0, 1].set_ylabel("Sharpe Ratio", fontsize=10)
        axes[0, 1].set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=43)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        ####Max Drawdown is plotted here after getting retrieved.
        axes[1, 0].bar(names, max_dd, color=self.colors[:len(names)])
        axes[1, 0].set_ylabel("Max Drawdown ($)", fontsize=10)
        axes[1, 0].set_title("Maximum Drawdown", fontsize=12, fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=42)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        ###fill Rate is§ plotted here after getting retrieved.
        axes[1, 1].bar(names, fill_rate, color=self.colors[:len(names)])
        axes[1, 1].set_ylabel("Fill Rate (%)", fontsize=10)
        axes[1, 1].set_title("Fill Rate", fontsize=11, fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=41)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        #setting layout and showing the plot
        plt.tight_layout()
        plt.show()
    
    def plot_order_flow_imbalance(
        self,
        time: np.ndarray,
        ofi: np.ndarray,
        price: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        this method sets up the plotting for order flow imbalance(OFI) microstructure.
        """
        if price is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            
            # OFI is being plotted here after getting retrieved
            ax1.plot(time, ofi, color='blue', linewidth=1.5, label='OFI')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.4)
            ax1.set_ylabel("OrderFlow Imbalance", fontsize=10)
            ax1.set_title("Order flow Imbalance", fontsize=13, fontweight="bold")
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3)
            
            # the pririce plotting here after getting retrieved
            ax2.plot(time, price, color='red', linewidth=1.5, label='Price')
            ax2.set_xlabel("Time (seconds)", fontsize=10)
            ax2.set_ylabel("Price", fontsize=10)
            ax2.set_title("Price Evolution", fontsize=12, fontweight="bold")
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(time, ofi, color='blue', linewidth=1.5, label='OFI')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel("Order flow Imbalance", fontsize=10)
            ax.set_title("Order Flow Imbalance", fontsize=10, fontweight="bold")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        
        # settin layout =>  showing the plot afterwards 
        plt.tight_layout()
        plt.show()
    
    def plot_intensity_functions(
        self,
        strategy: any,
        time_range: Tuple[float, float],
        num_points: int = 1000,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        this method plots Hawkes intensity funcs over time horison we are interested in.
    
        """
        # checking if any of our strat have any intensity in tem  
        time_points = np.linspace(time_range[0], time_range[1], num_points)
        # getting both buy and sell intensities  here 
        buy_intensities = [strategy.buy_intensity(t) for t in time_points] #buy
        sell_intensities = [strategy.sell_intensity(t) for t in time_points] #sell intensity 
        # setting subplotting 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # plotting buy intensity here after getting retrieved and recalled
        ax1.plot(time_points, buy_intensities, color='green', linewidth=2, label='Buy Intensity')
        ax1.set_ylabel("λ_buy(t)", fontsize=10)
        ax1.set_title("Buy intensity", fontsize=12, fontweight="bold")
        ax1.legend(loc="best") #setting the legend to best place here 
        ax1.grid(True, alpha=0.3)
        
        # plotting sell intensity here after getting retrieved and recalled
        ax2.plot(time_points, sell_intensities, color='red', linewidth=2, label='Sell Intensity')
        ax2.set_xlabel("Time (seconds)", fontsize=10)
        ax2.set_ylabel("λ_sell(t)", fontsize=10)
        ax2.set_title("Sell Intensity", fontsize=12, fontweight="bold")
        ax2.legend(loc="best")  #setting the legend to best place here 
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_spread_dynamics(
        self,
        quote_history: List[Dict],
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        plot the bid-ask spread dynamics of the model.
        """
        if not quote_history: return #no quote history to return so  exit the routine
        
        # retreive all the needed values here from the qote history if available 
        #via list compreh ensions
        times = [q['time'] for q in quote_history] #time vals 
        bids = [q['bid'] for q in quote_history] # bids vals
        asks = [q['ask'] for q in quote_history] #ask vals
        mids = [q['mid'] for q in quote_history] #mid vals here
        spreads = [q['ask'] -q['bid'] for q in quote_history] 
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # the qoutes plotting here after getting retrieved
        ax1.plot(times, bids, color='green', label='Bid', linewidth=1, alpha=0.7)
        ax1.plot(times, asks, color='red', label='Ask', linewidth=1, alpha=0.7)
        ax1.plot(times, mids, color='blue', label='Mid', linewidth=1.5, alpha=0.8)
        ax1.fill_between(times, bids, asks, alpha=0.2, color='gray')
        ax1.set_ylabel("Price", fontsize=10)
        ax1.set_title("Bid-Ask Quotes", fontsize=12, fontweight="bold")
        ax1.legend(loc="best") #setting the legend to best place here
        ax1.grid(True, alpha=0.3) #the apha vis value 
        
        # and the spread plotting here after having retreived iot
        ax2.plot(times, spreads, color='purple', linewidth=1.5)
        ax2.set_xlabel("Time (seconds)", fontsize=10)
        ax2.set_ylabel("Spread", fontsize=10)
        #axis title to ask bid speard 
        ax2.set_title("Bid-Ask Spread", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3) #alpha value for the grid vis
        #setting layout and then showing the plotting via plt.show()
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_comparison(
        self,
        params_dict: Dict[str, Dict],
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        compare calibrated params acorss models all 
        """
        # retreieve the common parameters here as a set in order to ensure there aree no duplicate values since sets doesnt allow duplication. 
        param_names = set()
        #interating through the params dict vals here => and update the params 
        for params in params_dict.values():
            param_names.update(params.keys())
        
        param_names = sorted(param_names)
        
        # create comp table here using an array 
        data = []
        # iterating through the params dict items here 
        for model_name, params in params_dict.items():
            # retrieveing the row values here for plotting purposes 
            row = [params.get(p, np.nan) for p in param_names]
            data.append(row)
        
        df = pd.DataFrame(data, columns=param_names, index=params_dict.keys())
        
        # plotting heatmap here using seaborn lib 
        fig, ax = plt.subplots(figsize=figsize) 
        #setting the heapmap hyperparam vals here 
        sns.heatmap(df, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title("Calibrated Parameters Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Parameters", fontsize=12)
        ax.set_ylabel("Models", fontsize=12)
        
        plt.tight_layout()
        plt.show()