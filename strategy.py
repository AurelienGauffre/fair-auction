import numpy as np

from mip import Model, xsum, minimize, BINARY, maximize


class Strategy():
    def __init__(self,v_k,d_k,B):
        self.v_k = v_k
        self.d_k = d_k
        self.B = B
        self.M = len(v_k)
        assert len(v_k) == len(d_k)
        self.utility = 0
        self.selected = []
    def report(self):
        print(self.utility)
        print(self.selected)

class HindsightStrat(Strategy):
    def __init__(self,v_k,d_k,B):
        super().__init__(v_k,d_k,B)
        self.solve()


    def solve(self):

        utility_k = [self.v_k[j] - self.d_k[j] for j in range(self.M)]
        model = Model("knapsack")
        model.verbose = 0
        x = [model.add_var(var_type=BINARY) for j in range(self.M)]
        model.objective = maximize(xsum(utility_k[i] * x[i] for i in range(self.M)))
        model += xsum(self.d_k[i] * x[i] for i in range(self.M)) <= self.B

        model.optimize()
        self.selected = [i for i in range(self.M) if x[i].x >= 0.99]

        self.utility = sum([self.v_k[el]-self.d_k[el] for el in self.selected])




class AdaptivePacingStrat(Strategy):
    def __init__(self,v_k,d_k,B):
        super().__init__(v_k,d_k,B)
        self.bids = np.zeros(self.M)
        self.mu = np.zeros(self.M)
        self.eps = (self.M) ** -0.5
        self.Budgets = np.zeros(self.M)
        self.Budgets[0] = self.B
        self.target_expenditure = self.B/self.M

        self.solve()


    def solve(self):

        mu_bar =  10
        for j in range(self.M):

            self.bids[j] = min(self.v_k[j] / (1 + self.mu[j]), self.Budgets[j])
            win = self.bids[j]>=self.d_k[j]
            if win:
                self.selected += [j]
            if j < self.M - 1:

                self.Budgets[j + 1] = self.Budgets[j]- self.d_k[j]*win

                self.mu[j + 1] = min(max(self.mu[j] - self.eps * (self.target_expenditure - self.d_k[j]*win), 0),
                                 mu_bar)


        self.utility = sum([self.v_k[el] - self.d_k[el] for el in self.selected])


class TruthfullBiddingStrat(Strategy):
    def __init__(self,v_k,d_k,B):
        super().__init__(v_k,d_k,B)
        self.bids = np.zeros(self.M)
        self.Budgets = np.zeros(self.M)
        self.Budgets[0] = self.B


        self.solve()


    def solve(self):
        for j in range(self.M):
            self.bids[j] = min(self.v_k[j], self.Budgets[j])
            win = self.bids[j]>=self.d_k[j]
            if win:
                self.selected += [j]
            if j < self.M - 1:

                self.Budgets[j + 1] = self.Budgets[j]- self.d_k[j]*win

        self.utility = sum([self.v_k[el] - self.d_k[el] for el in self.selected])
