import numpy as np
import pandas as pd

from mechanism import Mechanism,MultiplicativePacing,AdaptivePacing
from strategy import HindsightStrat,AdaptivePacingStrat,TruthfullBiddingStrat,AdaptivePacingStrat_cumulative
from bidders import Bidders, Bidders_budget
from auction import Auction
from fairness_contraint import Budget_proportion_constraint,Proportion_constraint,Proportion_constraint_cumulative

# plotly is an interactive and light library for plotting in python
import plotly.express as px
import plotly.graph_objects as go

def simulation_one_parameter(params, strats_type, constraints_type,n,plot=True):
    """
    Make n simulation of auction (Balseiro type) looping on a parameter. The parameter x to loop on has simply to be
    provided as a range, list or array and will automatically detected and called x_name
    strat_type : dictionnary of type of strats of the form {'AP':AdaptivePacingStrat,'AP fair':AdaptivePacingStrat}
    constraints_type : dictionnary of type of strats of the form {'AP':None,'AP fair':Proportion_constraint}
    n : total number of simulation. There will be n//len(x) simulation per value of x.

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
    copy_param_x = params[x_name] #create a copy of the x param before it is overwritten
    for i, x in enumerate(params[x_name]):
        print(x)
        for j in range(n_x):
            params[x_name]=x

            user = [0, 1] * (params['M'] // 2)

            B = params['M'] * params['expenditure']
            values = np.random.random((params['M'], params['N']))
            v_k = values[:, 0]
            v_k[::2] *= 1 / params['bias']
            v_k[1::2] *= params['bias']
            values = np.delete(values, 0, 1)
            d_k = np.max(values, axis=1)

            # instantiating constraints
            for name,const_type in constraints_type.items():
                if const_type is not None:
                    constraints[name]=const_type(params['lambda'])
                else :
                    constraints[name]=None
            # instantiating strats
            for name,strat_type in strats_type.items():
                strats[name]=strat_type(v_k,d_k,B,eps=params['eps'],eps_gamma=params['eps_gamma'],user=user,fairness_constraint=constraints[name])

            for name,strat in strats.items():
                df.iloc[k, :] = [params[x_name], strat.utility, strat.compute_fairness(),name]
                k += 1

    if plot:
        #Plot of fairness

        fig = px.box(df, x=x_name, y="fairness", color="name")
        fig.update_layout(title=f" Fairness  vs {x_name}")
        fig.show()

        #Plot of utility
        fig = px.box(df, x=x_name, y="utility", color="name")
        fig.update_layout(title=f" Utility  vs {x_name}")
        fig.show()

        # Plot of the ROC curve
        df['utility'] = df['utility'].astype('float')
        df['fairness'] = df['fairness'].astype('float')
        grouped_df = df.groupby([x_name , 'name']).mean()
        grouped_df['name'] = [grouped_df.index[el][1] for el in range(len(grouped_df))] #transform an index into columns
        fig = px.scatter(grouped_df, x='utility', y='fairness', color='name')
        fig.update_layout(title=f" Utility VS Fairness  for {x_name} between [{copy_param_x[0]},{copy_param_x[-1]}]")

        fig.show()


    return df,fig

