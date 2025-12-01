"""

this module implements multivariates Hawkes processes with both exponential and power-law kernels.
Based on my paper "Modelling Order Flow Asymmetries with Hawkes Processes".
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, List


class MultivariateHawkesCalibrator:
    """
    the class is used for calibrating multivariate hawkes process 
    """
    
    def __init__(self): #simple init meth 
        self.params = None
        self.log_likelihood = None
        self.kernel_type = None
        
    def calibrate_exponential_kernel(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        initial_guess: Optional[Dict] = None
    ) -> Dict:
        """
        this method calibrate bivariate Hawkes with exponential kernel.
        
        """
        # all the necessary params for the optimiser 
        T = event_times[-1]
        N = len(event_times)
        N_buy = np.sum(event_types == 1)
        N_sell = np.sum(event_types == 0)
        
        # init guess? if none => use default vals 
        if initial_guess is None:
            mu_buy_init = N_buy / T
            mu_sell_init = N_sell / T
            alpha_init = 0.2
            beta_init = 1.0
        else: #use init guess params given
            mu_buy_init = initial_guess.get("mu_buy", N_buy / T)
            mu_sell_init = initial_guess.get("mu_sell", N_sell / T)
            alpha_init = initial_guess.get("alpha", 0.2)
            beta_init = initial_guess.get("beta", 1.0)
        
        x0 = np.array([ # the init param giess matrix 
            mu_buy_init, mu_sell_init,
            alpha_init, alpha_init, alpha_init, alpha_init,  # α_bb, α_ss, α_bs, α_sb
            beta_init
        ])
        
        # the bounds that are used are as follows : μ > 0, 0 ≤ α < 1, β > 0
        bounds = [
            (1e-6, None), (1e-6, None),  #for both baseline  μ_buy, μ_sell
            (0, 0.99), (0, 0.99), (0, 0.99), (0, 0.99),  # checking excitation alphas
            (1e-6, None)  # and the decay param beta
        ]
        
        # here we are performing optimosation work 
        result = minimize(
            lambda x: -self._log_likelihood_exponential(x, event_times, event_types, T),
            x0,
            method="L-BFGS-B", #using L-BFGS-B "bounded optim"/!!
            bounds=bounds #the bounds
        )
        # we extract the params via the matrix result.x 
        mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta = result.x
        
        self.params = {
            "mu_buy": mu_buy,
            "mu_sell": mu_sell,
            "alpha_bb": alpha_bb,
            "alpha_ss": alpha_ss,
            "alpha_bs": alpha_bs,
            "alpha_sb": alpha_sb,
            "beta": beta
        }
        # turning it into a cost function 
        self.log_likelihood = -result.fun
        self.kernel_type = "exponential" #exponential kernel here 
        
        return self.params
    
    def calibrate_powerlaw_kernel(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        initial_guess: Optional[Dict] = None
    ) -> Dict:
        """
        calibrate multivariate Hawkes with power-law kernel.
    
        """
        # all the necessary params for the optimiser
        T = event_times[-1]
        N = len(event_times)
        N_buy = np.sum(event_types == 1)
        N_sell = np.sum(event_types == 0)
        
        # init guess? if none => use default vals
        if initial_guess is None:
            mu_buy_init = N_buy / T
            mu_sell_init = N_sell / T
            alpha_init = 0.2
            beta_init = 2.0  # β > 1 for power-law kernel here 
            c_init = 1.0 # scale param c. It cannot be negarive 
        else: #use init guess params given all 
            mu_buy_init = initial_guess.get("mu_buy", N_buy / T)
            mu_sell_init = initial_guess.get("mu_sell", N_sell / T)
            alpha_init = initial_guess.get("alpha", 0.2)
            beta_init = initial_guess.get("beta", 2.0)
            c_init = initial_guess.get("c", 1.0)
        # the init guess matrix used as an np arrau 
        x0 = np.array([
            mu_buy_init, mu_sell_init,
            # α_bb, α_ss, α_bs, α_sb (the excitation params cross and self both)
            alpha_init, alpha_init, alpha_init, alpha_init,  
            beta_init, c_init # scale and decay params that are used 
        ])
        
        # the required bounds with such vals : μ > 0, 0 ≤ α < 1, β > 1, c > 0
        bounds = [
            (1e-6, None), (1e-6, None),  # μ_buy, μ_sell
            (0, 0.99), (0, 0.99), (0, 0.99), (0, 0.99),  # alphas
            (1.01, 10),  # beta (must be > 1 so that we can actually integrate)
            (0.01, 10)  # c param that must be pos (not neg)
        ]
        
        # optimise the params here (minimise the neg log likelihod)
        result = minimize(
            lambda x: -self._log_likelihood_powerlaw(x, event_times, event_types, T),
            x0,
            method="L-BFGS-B", #using L-BFGS-B "bounded optim"/!!
            bounds=bounds #the bounds
        )
        # we extract the params via the matrix result.x
        mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta, c = result.x
        # setting the params here and storing them in a dict (hashgmap)
        self.params = {
            "mu_buy": mu_buy,
            "mu_sell": mu_sell,
            "alpha_bb": alpha_bb,
            "alpha_ss": alpha_ss,
            "alpha_bs": alpha_bs,
            "alpha_sb": alpha_sb,
            "beta": beta,
            "c": c
        }
        # turning it into a cost function
        self.log_likelihood = -result.fun
        self.kernel_type = "powerlaw" #powerlaw kernel here
        
        return self.params
    
    def _log_likelihood_exponential(
        self,
        params: np.ndarray,
        event_times: np.ndarray,
        event_types: np.ndarray,
        T: float
    ) -> float:
        """
        Log-likelihood for exponential kernel. FORMULA: L = Σ log(λ_k(tᵢ)) - ∫₀ᵀ [λ_buy(t) + λ_sell(t)] dt
        """
        # extract params from the matrix 
        mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta = params
        
        # ensure that the params are indeed valid 
        if mu_buy <= 0 or mu_sell <= 0 or beta <= 0:
            return -np.inf #iof not valid => return -inf
        if alpha_bb < 0 or alpha_ss < 0 or alpha_bs < 0 or alpha_sb < 0:
            return -np.inf #if not valid => return -inf
        if alpha_bb + alpha_sb >= 1 or alpha_ss + alpha_bs >= 1:
            return -np.inf #if not valid => return -inf
        
        N = len(event_times)
        log_intensity_sum = 0.0
        
        # calc log intensities at event times here. Complex foor loop maybe fix later???
        for i in range(N):
            # the time and type of event used 
            t = event_times[i]
            event_type = event_types[i]
            
            if event_type == 1:  # 1 indicated buy event
                intensity = mu_buy # baseline buy intensity
                for j in range(i):
                    dt = t - event_times[j] #time diff (dt mathching the paper)
                    if event_types[j] == 1:  # prev buy event
                        intensity += alpha_bb * beta * np.exp(-beta * dt)
                    else:  # prev sell event arrival
                        # intensity is now increment on account of to sell event
                        intensity+= alpha_sb * beta * np.exp(-beta * dt)
            else:  # sell event (0)
                intensity = mu_sell # baseline sell intensity
                for j in range(i):
                    dt = t - event_times[j] #time diff (dt matching the paper)
                    if event_types[j] == 1:  # prev buy
                        # intensity incremented on account of the buy event 
                        intensity +=alpha_bs *beta *np.exp(-beta * dt)
                    else:  # prev sell
                        intensity += alpha_ss * beta * np.exp(-beta * dt)
            
            if intensity <= 0: return -np.inf #invalud intensity so => return -inf
            # we are adding log intensity to the sum here 
            log_intensity_sum += np.log(intensity)
        
        # compensator (reinman integral)
        N_buy = np.sum(event_types == 1) #num of buy events
        N_sell = np.sum(event_types == 0) #num of sell events
        #compensator formula here similar to the math in the papper 
        compensator = (
            mu_buy * T + alpha_bb * N_buy + alpha_sb * N_sell +
            mu_sell * T + alpha_bs * N_buy + alpha_ss * N_sell
        )
        # log likelihood final calc here
        log_likelihood =log_intensity_sum- compensator
        
        return log_likelihood
    
    def _log_likelihood_powerlaw(
        self,
        params: np.ndarray,
        event_times: np.ndarray,
        event_types: np.ndarray,
        T: float
    ) -> float:
        """
        log-likelihood for power-law kernel.
        """
        mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta, c = params
        
        # checking if the params are indeed valud if not we spit out inf (-)
        if mu_buy <= 0 or mu_sell <= 0 or beta <= 1 or c <= 0:
            return -np.inf
        if alpha_bb < 0 or alpha_ss < 0 or alpha_bs < 0 or alpha_sb < 0:
            return -np.inf # return -inf if  in valid
        if alpha_bb + alpha_sb >= 1 or alpha_ss + alpha_bs >= 1:
            return -np.inf #same logic here 
        
        N = len(event_times)
        log_intensity_sum = 0.0
        
        # calcs log intensities at event times
        for i in range(N):
            t = event_times[i]
            event_type = event_types[i]
            
            if event_type == 1:  # this is the buy event
                intensity = mu_buy
                for j in range(i):
                    dt = t - event_times[j]
                    if event_types[j] == 1:  # this is the prev  buy event
                        # intensity incremented on account of the buy event
                        intensity += alpha_bb * beta * (dt + c) ** (-1 - beta)
                    else:  # this is prev  sell event
                        intensity += alpha_sb * beta * (dt + c) ** (-1 - beta)
            else:  # sell event here (branch for sell event)
                intensity = mu_sell 
                for j in range(i):
                    dt = t - event_times[j] # time diff(dt matching paper)
                    if event_types[j] == 1:  # prev buy event arrival
                        intensity += alpha_bs * beta * (dt + c) ** (-1 - beta)
                    else:  # prev sell event arrival 
                        # intensity incremented on account of the sell event
                        intensity += alpha_ss * beta * (dt + c) ** (-1 - beta)
            
            if intensity <= 0: return -np.inf # invalid intensity so => return -inf
            # we are adding log intensity to the sum here
            
            log_intensity_sum += np.log(intensity)
        
        # copmensator (the reimman integral) with power-law kernel integrated 
        N_buy = np.sum(event_types == 1) #num of buy events 
        N_sell = np.sum(event_types == 0) #num of sell events
        
        # integral of the power-law kernel from each event up to the horizon T
        integral_sum_bb = 0.0
        integral_sum_ss = 0.0
        integral_sum_bs = 0.0
        integral_sum_sb = 0.0
        # we are here integrating from each event time to the horizon T using a for loop
        for i in range(N):
            remaining_time = T - event_times[i]
            integral_contrib = (c ** (-beta) - (remaining_time + c) ** (-beta)) / beta
            
            if event_types[i] == 1:  # this is a buy event
                # integration logic here
                integral_sum_bb += alpha_bb * integral_contrib 
                integral_sum_bs += alpha_bs * integral_contrib
            else:  # this is a sale arrival 
                #integration logic here
                integral_sum_sb += alpha_sb * integral_contrib
                integral_sum_ss += alpha_ss * integral_contrib
        
        # final compensator calc is done here
        compensator = (
            mu_buy * T + integral_sum_bb +integral_sum_sb +mu_sell * T +integral_sum_bs +integral_sum_ss) #end of calc 
        
        log_likelihood = log_intensity_sum -compensator #log likelihood final calc here
        
        return log_likelihood

class HawkesMarketMaker:
    """
    MM maker using multivariate Hawkes processes.
    """
    # init meth 
    def __init__(
        self,
        params: Dict,
        kernel_type: str = "exponential",
        risk_aversion: float = 0.1,
        inventory_target: float = 0.0,
        imbalance_sensitivity: float = 0.5
    ):
        """
        init the market maker logic.
    
        """
        #specific params set here
        self.params = params
        self.kernel_type = kernel_type
        self.risk_aversion = risk_aversion
        self.inventory_target = inventory_target
        self.imbalance_sensitivity = imbalance_sensitivity
        
        self.buy_times = [] #array to store buy event times
        self.sell_times = [] #array to store sell event times
        
    def buy_intensity(self, t: float) -> float:
        """calcs buy intensity at time t."""
        mu_buy = self.params["mu_buy"] # baseline buy intensity
        alpha_bb = self.params["alpha_bb"] # self-excitation for buys
        alpha_sb = self.params["alpha_sb"] # cross-excitation: sells => buys
        beta = self.params["beta"] # decay param
        
        intensity = mu_buy # baseline buy intensity
        
        # recent events up to 100 secs 
        lookback = 100.0
        
        if self.kernel_type == "exponential": # exponential kernel case here 
            for ti in self.buy_times: # looping over buy event times 
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos 
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the exponential kernel (decay)
                    contribution = alpha_bb * beta * np.exp(-beta * dt)
                    # adding contribution to intensity so create the model 
                    intensity += contribution
            
            for ti in self.sell_times: # looping over sell event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the exponential kernel (decay)
                    contribution = alpha_sb * beta * np.exp(-beta * dt) 
                    # adding contribution to intensity so create the model
                    intensity += contribution 
        
        elif self.kernel_type == "powerlaw": # power-law kernel case here
            c = self.params["c"] # scale param for power-law kernel it must be postive
            for ti in self.buy_times: # looping over buy event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos 
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the power-law kernel (decay)
                    contribution = alpha_bb * beta * (dt + c) ** (-1 - beta)
                    # adding contribution to intensity so create the model
                    intensity += contribution
            
            for ti in self.sell_times: # looping over sell event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the power-law kernel (decay)
                    contribution = alpha_sb * beta * (dt + c) ** (-1 - beta)
                    # adding contribution to intensity so create the model
                    intensity += contribution
        
        # enssure that intensity is udner control (so cap it)
        return min(intensity, mu_buy * 10)
    
    def sell_intensity(self, t: float) -> float:
        """cals sell intensity at time t."""
        mu_sell = self.params["mu_sell"] # baseline sell intensity
        alpha_ss = self.params["alpha_ss"] # self-excitation param for sells
        alpha_bs = self.params["alpha_bs"] # cross-excitation: buys that has an impact on => sells
        beta = self.params["beta"] # decay param is set 
        
        intensity = mu_sell # the baseline intensity for sale
        
        # recent events (max 100 sec)
        lookback = 100.0
        
        if self.kernel_type == "exponential": # exponential kernel case here
            for ti in self.buy_times: # looping over buy event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the exponential kernel (decay)
                    contribution = alpha_bs * beta * np.exp(-beta * dt)
                    # adding contribution to intensity so create the model
                    intensity += contribution
            
            for ti in self.sell_times: # looping over sell event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the exponential kernel (decay)
                    contribution = alpha_ss * beta * np.exp(-beta * dt)
                    # adding contribution to intensity so create the model
                    intensity += contribution
        
        elif self.kernel_type == "powerlaw": # power-law kernel case here
            c = self.params["c"] # scale param for power-law kernel it must be postive
            for ti in self.buy_times: # looping over buy event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos
                    dt = t - ti #time diff (dt mathching the paper)
                    # contribution calc here using the power-law kernel (decay)
                    contribution = alpha_bs * beta * (dt + c) ** (-1 - beta)   
                    # adding contribution to intensity so create the model 
                    intensity += contribution
            
            for ti in self.sell_times: # looping over sell event times
                if ti < t and (t - ti) < lookback: # and checking a lookback sliding windwos
                    dt = t - ti     #time diff (dt mathching the paper)
                    # contribution calc here using the power-law kernel (decay)
                    contribution = alpha_ss * beta * (dt + c) ** (-1 - beta)
                    # adding contribution to intensity so create the model
                    intensity += contribution
        
        # enssure that intensity is udner control (so cap it)
        return min(intensity, mu_sell * 10)
    
    def order_flow_imbalance(self, t: float) -> float:
        """
        calc  (OFI).
        """
        lambda_buy = self.buy_intensity(t) # buy intensity at time t
        lambda_sell = self.sell_intensity(t) # sell intensity at time t
        # calc OFI here using the math formula 
        total_intensity = lambda_buy + lambda_sell
        if total_intensity < 1e-9: return 0.0 # avoid div by zero /0
        
        # OFI formula here
        ofi = (lambda_buy - lambda_sell) / total_intensity
        return ofi
    
    def get_quotes(
        self,
        t: float,
        mid_price: float,
        inventory: float
    ) -> Tuple[float, float]:
        """
        enerate bid and ask quotes exactly based on Hawkes intensities.
    
        """

        lambda_buy = self.buy_intensity(t) # buy intensity at time t
        lambda_sell = self.sell_intensity(t) # sell intensity at time t
        total_intensity = lambda_buy + lambda_sell # total intensity calc here
        
        # base spread calc here
        # Typical BTC spread is usually $5-50 depending on volatility
        min_spread = mid_price * 0.00005 # 0.5 bps min
        max_spread = mid_price * 0.0005 #5 bps max at most
        
        # intensity has an impact on our spread so the higher the intensity => tighter spread and we are using an inverse relation here
        intensity_factor = 1.0 / (1.0 + total_intensity * self.risk_aversion)
        base_spread = min_spread + (max_spread - min_spread) * intensity_factor
        
        # employ some penalty => increase(widen) spread if the inventory stays away from target we have
        inventory_penalty = mid_price *0.0001 *inventory # applying penalty here
        
        # if buy pressure is high => widen ask and then tighten bid
        ofi = self.order_flow_imbalance(t)
        ofi_adjustment = mid_price *0.00005 *ofi*self.imbalance_sensitivity
        
        # final bid and ask prices calc here
        bid_price = mid_price - base_spread / 2- inventory_penalty -ofi_adjustment
        ask_price = mid_price + base_spread / 2 -inventory_penalty +ofi_adjustment
        
        return bid_price, ask_price
    
    def record_event(self, t: float, event_type: str):
        """
        recording a new trade event in this method.
        """
        if event_type == "buy": # buy event
            self.buy_times.append(t) 
        else: # sell event # sell event
            self.sell_times.append(t)
        
        # clean old that are beyond 1 hour that we dont want /need 
        cutoff_time = t - 3600 # 1 hour ago
        # cleaning old events logc here
        self.buy_times = [ti for ti in self.buy_times if ti > cutoff_time]
        self.sell_times = [ti for ti in self.sell_times if ti > cutoff_time]
