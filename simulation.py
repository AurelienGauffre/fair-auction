import numpy as np
import pandas as pd

from mechanism import Mechanism,MultiplicativePacing,AdaptivePacing
from strategy import HindsightStrat,AdaptivePacingStrat,TruthfullBiddingStrat,AdaptivePacingStrat_bis
from bidders import Bidders, Bidders_budget
from auction import Auction
from fairness_contraint import Budget_proportion_constraint,Proportion_constraint,Proportion_constraint_bis

# plotly is an interactive and light library for plotting in python
import plotly.express as px
import plotly.graph_objects as go

def simulation_one_parameter(params, strats_type, constraints_type,n,plot=True):
    """
    Make n simulation of auction (Balseiro type) looping on a parameter
    strat_type : dictionnary of type of strats of the form {'AP':AdaptivePacingStrat,'AP fair':AdaptivePacingStrat}
    constraints_type : dictionnary of type of strats of the form {'AP':None,'AP fair':Proportion_constraint}
    """

    x_name = '' # Name of the parameter to loop on
    # for loop to detect on which parameter to loop automatically
    for param_name,param_value in params.items():
        if type(param_value) in [np.ndarray,list,range]:
            x_name = param_name
            print(x_name)
            break
    assert strats_type.keys() == constraints_type.keys()
    n_x = n // len(params[x_name]) # Number of auction per value of x_param
    n_strat = len(strats_type)
    strats = {}
    constraints = {}
    df = pd.DataFrame(columns=[x_name, 'utility', 'fairness', 'name'], index=range(n_strat * n))

    k = 0
    for i, x in enumerate(params[x_name]):
        for j in range(n_x):
            params[x_name]=x

            user = [0, 1] * (params['M'] // 2)

            B = params['M'] * 0.1
            values = np.random.random((params['M'], params['N']))
            v_k = values[:, 0]
            v_k[::2] *= 1 / params['bias']
            v_k[1::2] *= params['bias']
            values = np.delete(values, 0, 1)
            d_k = np.max(values, axis=1)

            # instantiating constraints
            for name,const_type in constraints_type.items():
                if const_type is not None:
                    constraints[name]=const_type(params['lams'])
                else :
                    constraints[name]=None
            # instantiating strats
            for name,strat_type in strats_type.items():
                strats[name]=strat_type(v_k,d_k,B,eps=params['eps'],eps_gamma=params['eps_gamma'],user=user,fairness_constraint=constraints[name])

            for name,strat in strats.items():
                df.iloc[k, :] = [params[x_name], strat.utility, strat.compute_fairness(),name]
                k += 1

    if plot:
        fig = px.box(df, x=x_name, y="fairness", color="name")
        fig.show()
        fig = px.box(df, x=x_name, y="utility", color="name")
        fig.show()
    return df

