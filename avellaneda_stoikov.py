"""
This module implements the classic Avellaneda-Stoikov model that is used for Market Making. In this paper it has been used as a baselines in a bid to compare the performance of Stochastic point processes (Hawkes) against.

"""

import numpy as np
from typing import Tuple, Optional, Self


class AvellanedaStoikovMM:
    """
    Avellaneda-Stoikov market making strategy.
    
    This is based off optimal market making under inventory rosk and asym infor 
    It uses Poisson based arrival rates to model order flows under the assumption that event arrivals are memoryless and indep. 
    """
    
    def __main__(
        cls,
        risk_aversion: float = 0.1,
        order_arrival_rate: float = 1.0,
        order_fill_probability: float = 0.5,
        inventory_target: float = 0.0,
        max_inventory: float = 10.0,
        volatility: float = 1.0,
        time_horizon: float = 1.0
    )-> Self:
        """
        init AS market maker.
        
        Parameters:
        -----------
        risk aversion : (γ)
        arrival of order flow rate : float (λ)
        order_fill_probability : float:  probability order getting filled (k)
        inventory_target : float: This is the target inventory level for the MM model
        max_inventory : float: the maximum allowed inventory to hold 
        volatility : float: volatility or stndard deviation (σ)
        time_horizon : float time horizon of convcern for trading (T)
        """
        self = super().__new__(cls)
        self.gamma = risk_aversion
        self.lambda_arrival = order_arrival_rate
        self.k = order_fill_probability
        self.inventory_target = inventory_target
        self.max_inventory = max_inventory
        self.sigma = volatility
        self.T = time_horizon
        return self
        
    def reservation_price(
        self,
        mid_price: float,
        inventory: float,
        time_remaining: float
    ) -> float:
        """
        reservation price computation using the following formula.
        r(t, q) = S - q * γ * σ² * (T - t)
        params :
        -----------
        mid_price : float S
        inventory : float= q
        time_remaining : float: T

        """
        reservation = mid_price - inventory * self.gamma * (self.sigma ** 2) * time_remaining
        return reservation
    
    def optimal_spread(
        self,
        time_remaining: float,
        inventory: float
    ) -> float:
        """
        Compute the ideal/ optimal bid-ask spread for AS MM model using the followg formula.
        
        δ = γ * σ² * (T - t) + (2 / γ) * ln(1 + γ / k)
        
        """
        # Base spread calculated from the AS formula written above
        spread = (
            self.gamma * (self.sigma ** 2) * time_remaining +
            (2.0 / self.gamma) * np.log(1 + self.gamma / self.k)
        )
        
        spread = max(spread, 0.01)  # this makes sure that spread is positive(+) n case in spits out negatuve (-) speard 
        
        return spread
    
    def get_quotes(
        self,
        t: float,
        mid_price: float,
        inventory: float,
        time_horizon: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Generate bid and ask quotes using the formula below and returns a tuple of bid and ask price.
        bid = r - δ/2
        ask = r + δ/2

        """
        T = time_horizon if time_horizon is not None else self.T
        time_remaining = max(T - t, 0.01)  # this is for safety so that we avoid division by zero
        
        # Reservation price calculation.
        r = self.reservation_price(mid_price, inventory, time_remaining)
        
        # Optimal spread based off calling our meth
        spread = self.optimal_spread(time_remaining, inventory)
        
        # calcualing bith bid and ask Quotes and finally returning them as a tuple
        bid_price = r - spread / 2
        ask_price = r + spread / 2
        
        return bid_price, ask_price # retirning the tuple 
    
    def fill_probability(
        self,
        quote_price: float,
        mid_price: float,
        side: str
    ) -> float:
        """
        calculates the nondeterministic (probabilistic) order fill.
        
        P(fill) = k * exp(-k * |quote - mid|)

        """
        distance = abs(quote_price - mid_price) # |quote - mid|
        prob = self.k * np.exp(-self.k * distance) # fill prob using an exponential decay function 
        return np.clip(prob, 0, 1) # clipping it to return values between 0 and 1 for prob 
    
    def expected_pnl(
        self,
        bid_price: float,
        ask_price: float,
        mid_price: float,
        inventory: float,
        time_remaining: float
    ) -> float:
        """
        Calculates the expected profit and loss (PnL) based on current quotes we have.
        """
        # probabilistic fills  probabilities
        p_bid = self.fill_probability(bid_price, mid_price, "bid") #bid fill prob
        p_ask = self.fill_probability(ask_price, mid_price, "ask") # ask fill probs
        
        # Expected immediate profit
        prof_imm = (
            p_ask * (ask_price - mid_price) +
            p_bid * (mid_price - bid_price)
        )
        
        # Risk penalty. the risk of having inventory over time and getting penalised as a result
        penal_for_rsk = 0.5 * self.gamma * (self.sigma ** 2) * (inventory ** 2) * time_remaining
        
        expected_pnl = prof_imm - penal_for_rsk # the pnl which is the immediate profit less the penality for risk
        
        return expected_pnl
    
    def update_volatility(
        self,
        price_history: np.ndarray,
        window: int = 100
    ):
        """
        This method updats the volatility estimate straight from  recent price evolution/history.
    
        """
        if len(price_history) < 2:
            return
        
        returns = np.diff(np.log(price_history[-window:]))  # calc logarithmic returns to create stationary series from the price wich is non-stationary
        
        # using std of log return to estimate probs 
        self.sigma = np.std(returns) if len(returns) > 0 else self.sigma
    
    def update_arrival_rate(
        self,
        trade_times: np.ndarray,
        window_seconds: float = 60.0
    ):
        """
        Update order arrival rate estimate.
        """
        if len(trade_times) < 2:
            return
        
        # Count trades in recent window
        recent_trades = trade_times[trade_times >= (trade_times[-1] - window_seconds)]
        self.lambda_arrival = len(recent_trades) / window_seconds


if __name__ == "__main__":
    
    mm = AvellanedaStoikovMM(
        risk_aversion=0.1,
        order_arrival_rate=1.0,
        order_fill_probability=0.5,
        volatility=2.0,
        time_horizon=100.0
    ) #init the AS MM model with params provided 
    
    # the code below simulated some trading activity. 
    mid_price = 100.0
    inventory = 0.0
    t = 0.0

    
    for t in [0, 25, 50, 75, 99]: # some timesteps for testing 
        bid, ask = mm.get_quotes(t, mid_price, inventory)
        spread = ask - bid
        
        # the following code simulate some inventory evolutions and changes over time 
        if t == 25: inventory = 5.0
        elif t == 50:
            inventory = -3.0 # changing inv levs 
        elif t == 75: inventory = 0.0
    t = 50
    for inv in [-5, -2, 0, 2, 5]: #inventory levls we can use for testing 
        bid, ask = mm.get_quotes(t, mid_price, inv)
        r = mm.reservation_price(mid_price, inv, mm.T - t)
    inventory = 0
    for t in [0, 25, 50, 75, 95, 99]: # and a few more time steps for testing 
        bid, ask = mm.get_quotes(t, mid_price, inventory)
        spread = ask - bid #calculating some spread 

