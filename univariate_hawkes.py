"""
Univariate Hawkes Process Implementation

Implements univariate Hawkes processes with exponential kernel for baseline comparison.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional


class UnivariateHawkesCalibrator:
    """
    Calibrates a univariate Hawkes process with exponential kernel.
    Intensity: λ(t) = μ + Σ α * β * exp(-β * (t - tᵢ)) for all events tᵢ < t
    """
    
    def __init__(self):
        self.params = None
        self.log_likelihood = None
        
    def calibrate(
        self, 
        event_times: np.ndarray,
        initial_guess: Optional[Dict] = None
    ) -> Dict:
        """
        Calibrate univariate Hawkes process using MLE.

        """
        T = event_times[-1]
        N = len(event_times)
        
        # initial guess for params. If it is set to noen => use default vals 
        if initial_guess is None:
            mu_init = N / T  # baseline intensity of the model 
            alpha_init = 0.3 #excitation param alpha 
            beta_init = 1.0 #decay rate beta
        else: #use initial guess provided
            mu_init = initial_guess.get("mu", N / T)
            alpha_init = initial_guess.get("alpha", 0.3)
            beta_init = initial_guess.get("beta", 1.0)
        
        x0 = np.array([mu_init, alpha_init, beta_init])
        
        # optimiser to perform MLE (EM algo)
        result = minimize(
            lambda x: -self._log_likelihood(x, event_times, T),
            x0,
            method="L-BFGS-B", # using L-BFGS-B "bounded optim"/
            bounds=[(1e-6, None), (0, 0.99), (1e-6, None)] #the bounds 
        )
        # extract the params 
        mu, alpha, beta = result.x
        self.params = {"mu": mu, "alpha": alpha, "beta": beta}
        self.log_likelihood = -result.fun
        
        return self.params
    
    def _log_likelihood(
        self, 
        params: np.ndarray, 
        event_times: np.ndarray, 
        T: float
    ) -> float:
        """
        calscs log-likelihood for univariate Hawkes model!!.
    
        """
        #the params are retreievedfrom the params arr 
        mu, alpha, beta = params
        
        # these are not valid params for Hawks so => return -inf
        if mu <= 0 or alpha < 0 or alpha >= 1 or beta <= 0:
            return -np.inf
        
        N = len(event_times)
        
        # summasstion of log intensities at event arrvival times times
        log_intensity_sum = 0.0
        #expensive dounle loop over events 
        for i in range(N):
            intensity = mu
            for j in range(i):
                dt = event_times[i] - event_times[j]
                intensity += alpha * beta * np.exp(-beta * dt)
            
            if intensity <= 0:  return -np.inf
            
            log_intensity_sum += np.log(intensity)
        
        # compensator (the reimman integral)
        compensator = mu * T + alpha * N
        #final log likelihood 
        log_likelihood=log_intensity_sum -compensator
        
        return log_likelihood
    
    def intensity(
        self, 
        t: float, 
        event_times: np.ndarray,
        params: Optional[Dict] = None
    ) -> float:
        """
        calc intensity at time t 
        """
        if params is None: params = self.params
        #if still none => we need to calibraye first 
        if params is None:
            raise ValueError("we dont have any parms yet.")
        
        mu = params["mu"]
        alpha = params["alpha"]
        beta = params["beta"]
        intensity = mu
        
        # Summasation of past events 
        past_events = event_times[event_times < t]
        #this is so that we get their contributions 
        for ti in past_events:
            dt = t - ti
            intensity += alpha * beta * np.exp(-beta * dt)
        
        return intensity


class UnivariateHawkesMarketMaker:
    """
    market maker using univariate Hawkes process.
    """
    
    def __init__(
        self,
        params: Dict,
        risk_aversion: float = 0.1,
        inventory_target: float = 0.0, #can also be changed here to get other vals when we call it mate
        max_inventory: float = 10.0
    ):
        """
        initialising the MM based upon the univariate Hawkes stochastic point processes.
        """
        #init all the params 
        self.params = params
        self.risk_aversion = risk_aversion
        self.inventory_target = inventory_target
        self.max_inventory = max_inventory
        self.event_times = []
        
    def intensity(self, t: float) -> float:
        """
        get curr intensity of the market base doff univar hawkes.
        """
        #retreieve all the oarams here!!
        mu = self.params["mu"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        
        #this is the curr intensity calculatted 
        intensity = mu
        #interating overa the entire past events 
        for ti in self.event_times:
            if ti < t: 
                dt = t-ti #the derivative time (little time)
                intensity+=alpha * beta * np.exp(-beta * dt) #using the math of Hawkes 
        
        return intensity
    
    def get_quotes(
        self, 
        t: float, 
        mid_price: float, 
        inventory: float
    ) -> Tuple[float, float]:
        """
        generating bid and ask quotes based upon curr intensity && inventory.
        """
        #getting the curr intensity here 
        current_intensity = self.intensity(t)
        
        # base spread. This is as we know is inversely proportional to intens
        current_intensity = np.clip(current_intensity, 0.1, 100.0)
        base_spread = self.risk_aversion * 100.0 / current_intensity  # applying much tighter spreads 
        
        # applying some degrees of inventory adjustment (skewing quotes in a bid to reduce inventory)
        inventory_skew = 0.5 * base_spread * (inventory - self.inventory_target)
        
        bid_price = mid_price - base_spread /2 - inventory_skew #bid price skewed calc by inventory 
        ask_price = mid_price + base_spread/2 - inventory_skew #same to ask price 
        
        return bid_price, ask_price #return the tuple 
    
    def record_event(self, t: float):
        """
        record a new trade event (1 hour max for much better computational efficiency).
        """
        self.event_times.append(t)
        
        # for computational complexity we avoid recording for over an hour
        if len(self.event_times) > 0:
            cutoff_time= t-3600  # a total of 1 hour ay mostt
            #just keep data within the hour 
            self.event_times =[x for x in self.event_times if x > cutoff_time]

