"""
Calibration Module

Implements Expectation-Maximization (EM) algorithm for Hawkes process calibration.
Provides alternative calibration methods beyond MLE.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize


class EMCalibrator:
    """
    Expectation-Maximization algorithm used to calibrate params of Hawkes process .
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        initialisers for EM (expectation maximisation) algo
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.params_history = [] # empty array to store param hist 
        
    def calibrate_univariate_em(
        self,
        event_times: np.ndarray,
        initial_params: Optional[Dict] = None
    ) -> Dict:
        """
        This method calibratyes univariate Hawkes model using EM algo.

        """
        T = event_times[-1] # the loval event tme
        N = len(event_times) # num events (time)
        
        # init params here 
        if initial_params is None:
            mu = N / T / 2
            alpha = 0.3 # just a guess for now 
            beta = 1.0
        else:
            mu = initial_params.get("mu", N / T / 2)
            alpha = initial_params.get("alpha", 0.3)
            beta = initial_params.get("beta", 1.0)
        
        for iteration in range(self.max_iterations):
            # E-step: the branching struct for events in grah like format
            p_background = np.zeros(N)  # Prob event is in bakground
            p_triggered = np.zeros((N, N))  # prob event i triggered event j (cross excitation)
            
            for i in range(N):
                t = event_times[i]
                
                # calcs intensity coming from everywhere pretty much
                lambda_bg = mu
                lambda_trig = 0.0 # set to 0 at this point (initially)
                
                for j in range(i):
                    dt = t - event_times[j]
                    # using an exponential kernel 
                    lambda_from_j = alpha *beta *np.exp(-beta * dt)
                    p_triggered[j, i] = lambda_from_j # the value is stored ere
                    lambda_trig += lambda_from_j # increm the aggregate trigger 
                
                lambda_total = lambda_bg + lambda_trig
                
                # the block below normalises probablities 
                if lambda_total > 0:
                    p_background[i] = lambda_bg / lambda_total # lamda total is demon and is above 0 
                    p_triggered[:, i] /= lambda_total
            
            # Maximasaion step: Updating params here
            mu_new = np.sum(p_background) / T
            alpha_new = np.sum(p_triggered) / N
            
            # Update the beta param via the solver
            numerator = 0.0
            denominator = 0.0
            # computationally exensive double for loop 
            for i in range(N):
                for j in range(i):
                    dt = event_times[i] - event_times[j]
                    numerator += p_triggered[j, i] * dt
                    denominator += p_triggered[j, i]
            
            if denominator > 0: # num stabiloty so we dont divide by 0
                beta_new = denominator / numerator
            else:
                beta_new = beta
            
            # the block below checks for convergenience
            param_change = abs(mu_new - mu) + abs(alpha_new - alpha) + abs(beta_new - beta)
            
            #and stores the new params in histort 
            mu, alpha, beta = mu_new, alpha_new, beta_new
            self.params_history.append({"mu": mu, "alpha": alpha, "beta": beta})
            
            if param_change < self.tolerance:
                print(f"EM converged in {iteration + 1} iterations")
                break
        
        return {"mu": mu, "alpha": alpha, "beta": beta}
    
    def calibrate_multivariate_em(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        initial_params: Optional[Dict] = None
    ) -> Dict:
        """
        Calibrate multivariate Hawkes params using EM algo.
        """
        T = event_times[-1]
        N = len(event_times)
        N_buy = np.sum(event_types == 1)
        N_sell = np.sum(event_types == 0)
        
        # init params here 
        if initial_params is None:
            mu_buy = N_buy / T / 2
            mu_sell = N_sell / T / 2
            alpha_bb = alpha_ss = alpha_bs = alpha_sb = 0.2
            beta = 1.0
        # else ensure that params are set from the dict 
        else:
            mu_buy = initial_params.get("mu_buy", N_buy / T / 2)
            mu_sell = initial_params.get("mu_sell", N_sell / T / 2)
            alpha_bb = initial_params.get("alpha_bb", 0.2)
            alpha_ss = initial_params.get("alpha_ss", 0.2)
            alpha_bs = initial_params.get("alpha_bs", 0.2)
            alpha_sb = initial_params.get("alpha_sb", 0.2)
            beta = initial_params.get("beta", 1.0)
        
        for iteration in range(self.max_iterations):
            # E-step: here we are computing branching probs
            p_background_buy = np.zeros(N)
            p_background_sell = np.zeros(N)
            ##[from an event => to an to_event and from a type => to a type]
            p_triggered = np.zeros((N, N, 2, 2))  
            
            for i in range(N):
                t = event_times[i]
                event_type_i = event_types[i]
                
                if event_type_i == 1:  # this event triggers (indicates) a buy
                    lambda_bg = mu_buy
                else:  # this triggers a sale event
                    lambda_bg = mu_sell
                
                lambda_trig = 0.0
                
                for j in range(i):
                    dt = t - event_times[j]
                    event_type_j = event_types[j]
                    
                    # looking at and determining what alpha to utilise 
                    if event_type_i == 1 and event_type_j == 1: # buy singals buy <- buy
                        alpha = alpha_bb
                    elif event_type_i == 0 and event_type_j == 0:  # sell signals sell  <- sell
                        alpha = alpha_ss
                    elif event_type_i == 0 and event_type_j == 1:  # cross excit sell <= buy
                        alpha = alpha_bs
                    else:  # buy <- sell (also cross excitation)
                        alpha = alpha_sb
                    # calculating the contrib from evemt j tall the way to event i
                    lambda_from_j = alpha * beta * np.exp(-beta * dt)
                    p_triggered[j, i, event_type_j, event_type_i] = lambda_from_j
                    lambda_trig += lambda_from_j
                #calculate total prob
                lambda_total = lambda_bg + lambda_trig
                
                # Normalise the probs to sum up to 1.
                if lambda_total > 0:
                    if event_type_i == 1:
                        # if thats a buy => store in buys
                        p_background_buy[i] = lambda_bg / lambda_total
                    else:
                        #else store in sells background
                        p_background_sell[i] = lambda_bg / lambda_total
                    # norm triggered probs 
                    p_triggered[:, i, :, :] /= lambda_total
            
            # Maximisation step: looking at updating parames via summations and normalusations
            mu_buy_new = np.sum(p_background_buy) / T
            mu_sell_new = np.sum(p_background_sell) / T
            
            # counting the  alphas that have come from cross excitations and self excitations both 
            count_bb = np.sum(p_triggered[:, :, 1, 1])
            count_ss = np.sum(p_triggered[:, :, 0, 0])
            count_bs = np.sum(p_triggered[:, :, 1, 0])
            count_sb = np.sum(p_triggered[:, :, 0, 1])
            # updating the alpha s
            alpha_bb_new = count_bb / N_buy if N_buy > 0 else alpha_bb
            alpha_ss_new = count_ss / N_sell if N_sell > 0 else alpha_ss
            alpha_bs_new = count_bs / N_buy if N_buy > 0 else alpha_bs
            alpha_sb_new = count_sb / N_sell if N_sell > 0 else alpha_sb
            
            # updating the beta params 
            numerator = 0.0
            denominator = 0.0
            # computationally exensive double for loop here 
            for i in range(N):
                for j in range(i):
                    dt = event_times[i] - event_times[j]
                    weight = np.sum(p_triggered[j, i, :, :])
                    numerator += weight * dt
                    denominator += weight
            
            if denominator > 0:
                beta_new = denominator / numerator
            else:
                beta_new = beta
            
            # Checking for convergence
            param_change = (
                abs(mu_buy_new - mu_buy) + abs(mu_sell_new - mu_sell) +
                abs(alpha_bb_new - alpha_bb) + abs(alpha_ss_new - alpha_ss) +
                abs(alpha_bs_new - alpha_bs) + abs(alpha_sb_new - alpha_sb) +
                abs(beta_new - beta)
            )
            
            # Update the mus all bot all events 
            mu_buy, mu_sell = mu_buy_new, mu_sell_new
            alpha_bb, alpha_ss = alpha_bb_new, alpha_ss_new
            alpha_bs, alpha_sb = alpha_bs_new, alpha_sb_new
            beta = beta_new
            # params dict 
            params = {
                "mu_buy": mu_buy, "mu_sell": mu_sell,
                "alpha_bb": alpha_bb, "alpha_ss": alpha_ss,
                "alpha_bs": alpha_bs, "alpha_sb": alpha_sb,
                "beta": beta
            }
            self.params_history.append(params.copy())
            
            if param_change < self.tolerance:
                print(f"EM converged in {iteration + 1} iterations")
                break
        
        return params
    
    def cross_validate(
        self,
        event_times: np.ndarray,
        event_types: Optional[np.ndarray] = None,
        n_folds: int = 5,
        method: str = "univariate"
    ) -> Tuple[Dict, float]:
        """
        This methd Performing k-fold cross-validation.
        """
        N = len(event_times)
        fold_size = N // n_folds # esure correct div
        
        log_likelihoods = [] # arrays to store lolg like and all the params 
        all_params = []
        
        for fold in range(n_folds):
            # Spliting the data here into train and test 
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else N
            # using a mask (bool mask) 
            train_mask = np.ones(N, dtype=bool)
            train_mask[test_start:test_end] = False
            # checking and splitting on meth 
            train_times = event_times[train_mask]
            test_times = event_times[~train_mask]
            
            if method == "univariate": #univar hawkes 
                # calib on da training set
                params = self.calibrate_univariate_em(train_times)
                
                # Evaluate on test set
                # log likeligood comp
                ll = self._compute_univariate_ll(test_times, params)
            
            else:  # multivariate hawkes 
                train_types = event_types[train_mask]
                test_types = event_types[~train_mask]
                #param uodates after em algo 
                params = self.calibrate_multivariate_em(train_times, train_types)
                ll = self._compute_multivariate_ll(test_times, 
                                                   test_types, params)
            # storeing the res in both arrays 
            log_likelihoods.append(ll)
            all_params.append(params)
        
        # returning the entire parameters from best fold after valid
        best_idx = np.argmax(log_likelihoods)
        best_params = all_params[best_idx]
        avg_ll = np.mean(log_likelihoods)
        
        return best_params, avg_ll
    
    def _compute_univariate_ll(
        self,
        event_times: np.ndarray,
        params: Dict
    ) -> float:
        """Compute log-likelihood for univariate Hawkes."""
        T = event_times[-1]
        N = len(event_times)
        mu = params["mu"]
        alpha = params["alpha"]
        beta = params["beta"]
        
        # computing the log likelihood here 
        log_sum = 0.0
        # double for loop for comp 
        for i in range(N):
            intensity = mu
            for j in range(i):
                dt = event_times[i] - event_times[j]
                #cjecking intensity 
                intensity += alpha * beta * np.exp(-beta * dt)
            
            if intensity > 0:
                log_sum += np.log(intensity) # if bigger than 0 => log it 
            else: # invalid si return inf 
                return -np.inf
        #compensator cals here 
        compensator = mu * T + alpha * N
        return log_sum - compensator
    
    def _compute_multivariate_ll(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        params: Dict
    ) -> float:
        """Compute log-likelihood for multivariate Hawkes."""
        T = event_times[-1]
        N = len(event_times)
        # computing the log likelihood here 
        log_sum = 0.0
        # double for loop for comp 
        for i in range(N):
            t = event_times[i]
            event_type = event_types[i] #retrieving the event type 1 => but and zero => sell 
            
            if event_type == 1:
                intensity = params["mu_buy"] # => store in buys 
                for j in range(i):
                    dt = t - event_times[j]
                    if event_types[j] == 1: # buy signal so buy 
                        intensity += params["alpha_bb"] * params["beta"] * np.exp(-params["beta"] * dt) # check intensity 
                    else: # sell signal so buy
                        intensity += params["alpha_sb"] * params["beta"] * np.exp(-params["beta"] * dt)
            else:
                intensity = params["mu_sell"] # => store in sells
                for j in range(i):
                    dt = t - event_times[j]
                    if event_types[j] == 1: # buy signal used so sell
                        intensity += params["alpha_bs"] * params["beta"] * np.exp(-params["beta"] * dt)
                    else: # sell signal used so sell here (ss cross excitation=)
                        intensity += params["alpha_ss"] * params["beta"] * np.exp(-params["beta"] * dt)
            
            if intensity > 0: #valud so log it 
                log_sum += np.log(intensity)
            else:
                return -np.inf # invalid so return inf
        #compensator cals here for differe events 
        
        N_buy = np.sum(event_types == 1) #sum up buys
        N_sell = np.sum(event_types == 0) #sum up sells
        
        compensator = (
            params["mu_buy"] * T + params["alpha_bb"] * N_buy + params["alpha_sb"] * N_sell +
            params["mu_sell"] * T + params["alpha_bs"] * N_buy + params["alpha_ss"] * N_sell
        )
        
        return log_sum - compensator


if __name__ == "__main__":
    print("EM Calibration Example just via using synthetic data for basic testing")
    
    # Generating synthetic random data from an exponential distr 
    np.random.seed(42)
    T = 1000 #time 
    # getting all the event types 
    event_times = np.sort(np.random.exponential(1.0, 500).cumsum())
    event_times = event_times[event_times < T]
    
    # Calibratation using EM algo 
    calibrator = EMCalibrator(max_iterations=50, tolerance=1e-4)
    params = calibrator.calibrate_univariate_em(event_times)
   
