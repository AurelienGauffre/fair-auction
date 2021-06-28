import numpy as np

from mip import Model, xsum, minimize, BINARY, maximize
from fairness_contraint import Proportion_constraint, Proportion_constraint_bis


class Strategy():
    def __init__(self, v_k, d_k, B,eps=None,eps_gamma=None,user=None, fairness_constraint=None):
        self.v_k = v_k
        self.d_k = d_k
        self.B = B
        self.M = len(v_k)
        self.eps = eps if eps is not None else self.M**(-.5)
        self.eps_gamma = eps_gamma if eps_gamma is not None else self.M ** (-.5)
        assert len(v_k) == len(d_k)
        self.utility = 0
        self.selected = []
        self.user = user
        if user is not None:
            assert len(user) == self.M
            self.Gamma = len(np.unique(user))

        self.fairness_constraint = fairness_constraint

    def report(self):
        # print(self.utility)
        # print(self.selected)

        if self.user is not None:
            print(sum([self.user[el] == 0 for el in self.selected]) / len(self.selected))
    def compute_fairness(self):
        return sum([self.user[el] == 0 for el in self.selected]) / len(self.selected) *2 #to adapt for more class
    def compute_utility(self):
        return sum([self.v_k[el] - self.d_k[el] for el in self.selected])

class HindsightStrat(Strategy):
    def __init__(self, v_k, d_k, B,eps=None,eps_gamma=None, user=None, fairness_constraint=None):
        super().__init__(v_k, d_k, B,eps,eps_gamma, user, fairness_constraint)
        self.solve()

    def solve(self):
        utility_k = [self.v_k[j] - self.d_k[j] for j in range(self.M)]

        model = Model("knapsack")

        model.verbose = 0

        x = [model.add_var(var_type=BINARY) for j in range(self.M)]
        model.objective = maximize(xsum(utility_k[i] * x[i] for i in range(self.M)))
        model += xsum(self.d_k[i] * x[i] for i in range(self.M)) <= self.B

        if type(self.fairness_constraint) == Proportion_constraint:
            for gamma in range(self.Gamma):
                # o_gamma = np.array(np.array(self.user)==gamma,dtype=float) # does not work with np array
                o_gamma = [self.user[i] == gamma for i in range(self.M)]

                u_gamma = sum(o_gamma) / self.M / (self.fairness_constraint.lam)

                model += u_gamma * xsum(x[i] for i in range(self.M)) >= xsum(o_gamma[i] * x[i] for i in range(self.M))
        model.optimize()

        # print(model.num_solutions)
        self.selected = [i for i in range(self.M) if x[i].x >= 0.99]

        self.utility = self.compute_utility()


class AdaptivePacingStrat(Strategy):
    def __init__(self, v_k, d_k, B,eps=None,eps_gamma=None, user=None, fairness_constraint=None):
        super().__init__(v_k, d_k, B,eps,eps_gamma, user, fairness_constraint)
        self.bids = np.zeros(self.M)
        self.mu = np.zeros(self.M)
        self.Budgets = np.zeros(self.M)
        self.Budgets[0] = self.B
        self.target_expenditure = self.B / self.M

        if type(self.fairness_constraint) in [Proportion_constraint, Proportion_constraint_bis]:
            self.mu_gamma = np.zeros((self.Gamma, self.M))
            self.o_gamma = np.zeros((self.Gamma, self.M))
            for gamma in range(self.Gamma):
                self.o_gamma[gamma, :] = [self.user[i] == gamma for i in range(self.M)]

        self.solve()

    def solve(self):

        mu_bar = 10

        if self.fairness_constraint is None:
            for j in range(self.M):
                self.bids[j] = min(self.v_k[j] / (1 + self.mu[j]), self.Budgets[j])
                win = self.bids[j] >= self.d_k[j]
                if win:
                    self.selected += [j]
                if j < self.M - 1:
                    self.Budgets[j + 1] = self.Budgets[j] - self.d_k[j] * win

                    self.mu[j + 1] = min(max(self.mu[j] - self.eps * (self.target_expenditure - self.d_k[j] * win), 0),
                                         mu_bar)

        elif type(self.fairness_constraint) == Proportion_constraint:
            for j in range(self.M):
                # a optimiser avec un produit scalaire
                # self.bids[j] = min((self.v_k[j] - sum(
                #     [self.mu_gamma[gamma, j] * (sum(self.o_gamma[gamma, :j]) - self.o_gamma[gamma, j] * (
                #             j + 1) * self.fairness_constraint.lam) for gamma in range(self.Gamma)]
                #
                # )) / (1 + self.mu[j]), self.Budgets[j])

                # self.bids[j] = min((self.v_k[j] - sum(
                #     [self.mu_gamma[gamma, j] * (self.o_gamma[gamma, j] * (j + 1) - sum(self.o_gamma[gamma, :j]))
                #      for gamma in range(self.Gamma)])) / (1 + self.mu[j]),
                #                    self.Budgets[j])

                self.bids[j] = min(
                    (self.v_k[j] - self.mu_gamma[self.user[j], j]) / (1 + self.mu[j]),
                    self.Budgets[j])

                win = self.bids[j] >= self.d_k[j]
                if win:
                    self.selected += [j]
                if j < self.M - 1:
                    self.Budgets[j + 1] = self.Budgets[j] - self.d_k[j] * win

                    self.mu[j + 1] = min(max(self.mu[j] - self.eps * (self.target_expenditure - self.d_k[j] * win), 0),
                                         mu_bar)

                    for gamma in range(self.Gamma):
                        # u_gamma=len(self.selected)*sum(self.user[0:j])/(j+1)**2/self.fairness_constraint.lam
                        # self.mu_gamma[gamma, j + 1] = min(
                        #     max(self.mu_gamma[gamma,j] - self.eps * (u_gamma- self.o_gamma[gamma,j] * win), 0),
                        #     mu_bar)

                        # self.mu_gamma[gamma, j + 1] = min(
                        #     max(self.mu_gamma[gamma, j] - self.eps_gamma * (
                        #             sum(self.o_gamma[gamma, :j]) - self.o_gamma[gamma, j] * (
                        #             j + 1) * self.fairness_constraint.lam) * win, 0),
                        #     mu_bar)

                        # Avec 1 approximation sur 2
                        # self.mu_gamma[gamma, j + 1] = min(
                        #     max(self.mu_gamma[gamma, j] - self.eps_gamma * (
                        #             np.sum(self.o_gamma[gamma, :j])/(3*(j+1)*self.fairness_constraint.lam) - self.o_gamma[gamma, j] * win), 0),
                        #     mu_bar)

                        #Avec 2 approximation sur 2
                        # self.mu_gamma[gamma, j + 1] = min(
                        #     max(self.mu_gamma[gamma, j] - self.eps_gamma * (
                        #             1/(3*(2)*self.fairness_constraint.lam) - self.o_gamma[gamma, j] * win), 0),
                        #     mu_bar)

                        self.mu_gamma[gamma, j + 1] = min(
                            max(self.mu_gamma[gamma, j] - self.eps_gamma * (0.5/self.fairness_constraint.lam - self.o_gamma[gamma, j] * win), 0),mu_bar)

        elif type(self.fairness_constraint) == Proportion_constraint_bis:
            for j in range(self.M):

                self.bids[j] = min(
                    (self.v_k[j]-self.mu_gamma[self.user[j], j]) / (1 + self.mu[j]  ),
                    self.Budgets[j])
                win = self.bids[j] >= self.d_k[j]
                if win:
                    self.selected += [j]
                if j < self.M - 1:
                    self.Budgets[j + 1] = self.Budgets[j] - self.d_k[j] * win
                    self.mu[j + 1] = min(max(self.mu[j] - self.eps * (self.target_expenditure - self.d_k[j] * win), 0),
                                         mu_bar)
                    for gamma in range(self.Gamma):
                        self.mu_gamma[gamma, j + 1] = min(max(self.mu_gamma[gamma, j] - self.eps_gamma * (
                                np.sum([self.o_gamma[gamma, :j]]) / (j + 1)
                                - self.fairness_constraint.lam *sum([self.o_gamma[gamma, el] for el in self.selected]) / (len(
                            self.selected) + 0.01)), 0), mu_bar)
        self.utility = self.compute_utility()


class AdaptivePacingStrat_bis(Strategy):
    def __init__(self, v_k, d_k, B,eps=None,eps_gamma=None,user=None, fairness_constraint=None):
        super().__init__(v_k, d_k, B,eps,eps_gamma, user, fairness_constraint)
        self.bids = np.zeros(self.M)
        self.mu = np.zeros(self.M)
        self.Budgets = np.zeros(self.M)
        self.Budgets[0] = self.B
        self.target_expenditure = self.B / self.M

        if type(self.fairness_constraint) in [Proportion_constraint, Proportion_constraint_bis]:
            self.mu_gamma = np.zeros((self.Gamma, self.M))
            self.o_gamma = np.zeros((self.Gamma, self.M))
            for gamma in range(self.Gamma):
                self.o_gamma[gamma, :] = [self.user[i] == gamma for i in range(self.M)]

        self.solve()

    def solve(self):

        mu_bar = 10

        if self.fairness_constraint is None:
            for j in range(self.M):
                self.bids[j] = min(self.v_k[j] / (1 + self.mu[j]), self.Budgets[j])
                win = self.bids[j] >= self.d_k[j]
                if win:
                    self.selected += [j]
                if j < self.M - 1:
                    self.Budgets[j + 1] = self.Budgets[j] - self.d_k[j] * win

                    self.mu[j + 1] = min(max(self.mu[j] - self.eps * (self.target_expenditure*(j+1) - sum([self.d_k[el] for el in self.selected])), 0),
                                         mu_bar)

        self.utility = self.compute_utility()

class TruthfullBiddingStrat(Strategy):
    def __init__(self, v_k, d_k, B):
        super().__init__(v_k, d_k, B)
        self.bids = np.zeros(self.M)
        self.Budgets = np.zeros(self.M)
        self.Budgets[0] = self.B

        self.solve()

    def solve(self):
        for j in range(self.M):
            self.bids[j] = min(self.v_k[j], self.Budgets[j])
            win = self.bids[j] >= self.d_k[j]
            if win:
                self.selected += [j]
            if j < self.M - 1:
                self.Budgets[j + 1] = self.Budgets[j] - self.d_k[j] * win

        self.utility = self.compute_utility()
