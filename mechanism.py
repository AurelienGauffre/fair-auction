import numpy as np

from fairness_contraint import Proportion_constraint, Budget_proportion_constraint
from utils import mask
from mip import Model, xsum, minimize, BINARY, maximize


class Mechanism():
    def __init__(self, bidders, users=None, fairness_constraint=None):
        '''
        Q: Allocation rule: Q[i][j] is the probability that i get j (each column is a distribution)
        P: Payment rule: P[i][j] expected payment by i for item j
        p: Vector of sizes M of the prices of the objects
        '''
        self.bidders = bidders
        self.users = np.array(users)
        self.Gamma = len(np.unique(users)) if users is not None else None
        self.fairness_constraint = fairness_constraint
        self.M = bidders.M
        self.N = bidders.N
        self.best_price = np.amax(self.bidders.values, axis=0)
        assert bidders.values is not None, 'Bidders have not revealed their object values.'
        self.Q = np.zeros((self.bidders.N, self.bidders.M))
        self.P = np.zeros((self.bidders.N, self.bidders.M))
        #self.prices = np.zeros(self.bidders.M)


        # self.T = 1

    def compute_Q(self):
        raise NotImplementedError

    def compute_P(self):
        raise NotImplementedError

    def compute_second_best_price(self):
        return [np.sort(self.bidders.values[:, j])[-2] for j in range(self.bidders.M)]

    def compute_revenue(self):
        return sum(self.prices)

    def compute_utility(self):
        return [sum([(self.bidders.values[i][j] - self.prices[j]) * self.Q[i][j] for j in range(self.M)]) for i in
                range(self.N)]

    def compute_social_welfare(self):
        return np.sum([[self.Q[i][j] * self.bidders.values[i][j] for j in range(self.M)] for i in range(self.N)])

    def compute_individual_utility(self):
        return sum([(self.bidders.values[0][j] - self.prices[j]) * self.Q[0][j] for j in range(self.M)])

    def report(self):
        print('B (budgets)\n', self.bidders.B)
        print('values\n', self.bidders.values)
        print('Q (allocation probability)\n', self.Q)
        print('P (payment)\n', self.P)
        print('p (price)\n', self.prices)
        print('total revenue\n', sum(self.prices))
        print('utility\n', self.compute_utility())
        print('social welfare\n', self.compute_social_welfare())

    def stats(self):
        self.total_revenue = sum(self.prices)
        self.utility = self.compute_utility()
        self.social_welfare = self.compute_social_welfare()


class VickereyAuction(Mechanism):
    def __init__(self, bidders):
        super().__init__(bidders)
        self.second_best_price = self.compute_second_best_price()
        self.compute_Q()
        self.compute_P()
        self.prices = self.compute_second_best_price()
        self.bids = self.bidders.values

    def compute_Q(self):
        for j in range(self.bidders.M):
            winners = np.array(np.argwhere(self.bidders.values[:, j] == self.best_price[j]).flatten().tolist())
            self.Q[winners, j] = 1 / len(winners)

    def compute_P(self):
        for j in range(self.bidders.M):
            winner = np.random.choice(range(self.bidders.N), p=self.Q[:, j])
            self.P[winner, j] = self.second_best_price[j]


class FirstPriceAuction(Mechanism):
    def __init__(self, bidders):
        super().__init__(bidders)
        self.compute_Q()
        self.compute_P()
        self.bids = self.bidders.values

    def compute_Q(self):
        for j in range(self.bidders.M):
            winners = np.array(np.argwhere(self.bidders.values[:, j] == self.best_price[j]).flatten().tolist())
            self.Q[winners, j] = 1 / len(winners)

    def compute_P(self):
        for j in range(self.bidders.M):
            winner = np.random.choice(range(self.bidders.N), p=self.Q[:, j])
            self.P[winner, j] = self.best_price[j]


class MultiplicativePacing(Mechanism):
    def __init__(self, bidders, users, fairness_constraint):
        super().__init__(bidders, users, fairness_constraint)

        self.alpha = None
        self.utility = None
        # assert isinstance(bidders,Bidders_budget),"Multiplicative Pacing mechanism needs budget" #Ne marche pas actuellement voir comment check des sous classes!
        self.solve()
        self.compute_Q()
        self.compute_P()
        self.stats()
        self.no_solution = False

    def solve(self):
        model = Model()
        model.verbose = 0
        if type(self.fairness_constraint) in [type(None), Proportion_constraint]:
            v_bar = [np.max(self.bidders.values[:, j]) for j in range(self.M)]

            d = [[model.add_var(var_type=BINARY) for j in range(self.M)] for i in range(self.N)]
            w = [[model.add_var(var_type=BINARY) for j in range(self.M)] for i in range(self.N)]
            r = [[model.add_var(var_type=BINARY) for j in range(self.M)] for i in range(self.N)]
            y = [model.add_var(var_type=BINARY) for i in range(self.N)]

            alpha = [model.add_var(lb=0, ub=1) for i in range(self.N)]

            p = [model.add_var() for j in range(self.M)]
            h = [model.add_var() for j in range(self.M)]
            s = [[model.add_var() for j in range(self.M)] for i in range(self.N)]

            model.objective = maximize(xsum(p[j] for j in range(self.M)))

            # CONSTRAINTS
            # 1
            for i in range(self.N):
                model += xsum(s[i][j] for j in range(self.M)) <= self.bidders.B[i]
            # 2
            for i in range(self.N):
                model += xsum(s[i][j] for j in range(self.M)) >= y[i] * self.bidders.B[i]
            # 3
            for i in range(self.N):
                model += alpha[i] >= 1 - y[i]
            # 4
            for j in range(self.M):
                model += xsum(s[i][j] for i in range(self.N)) == p[j]
            # 5
            for j in range(self.M):
                for i in range(self.N):
                    model += s[i][j] <= self.bidders.B[i] * d[i][j]
            # 6
            for j in range(self.M):
                for i in range(self.N):
                    model += h[j] >= alpha[i] * self.bidders.values[i][j]
            # 7
            for j in range(self.M):
                for i in range(self.N):
                    model += h[j] <= alpha[i] * self.bidders.values[i][j] + (1 - d[i][j]) * v_bar[j]
            # 8
            for j in range(self.M):
                for i in range(self.N):
                    model += w[i][j] <= d[i][j]
            # 9
            for j in range(self.M):
                for i in range(self.N):
                    model += p[j] >= alpha[i] * self.bidders.values[i][j] - w[i][j] * self.bidders.values[i][j]
            # 10
            for j in range(self.M):
                for i in range(self.N):
                    model += p[j] <= alpha[i] * self.bidders.values[i][j] + (1 - r[i][j]) * v_bar[j]
            # 11
            for j in range(self.M):
                model += xsum(w[i][j] for i in range(self.N)) == 1
            # 12
            for j in range(self.M):
                model += xsum(r[i][j] for i in range(self.N)) == 1
                # 13
            for j in range(self.M):
                for i in range(self.N):
                    model += r[i][j] + w[i][j] <= 1

            if type(self.fairness_constraint) == Proportion_constraint:
                l_1 = sum(self.users) / len(self.users)
                l_0 = 1 - l_1
                l_0 *= self.fairness_constraint.gamma
                l_1 *= self.fairness_constraint.gamma
                print('l_1', l_1)

                nb_win_i = [model.adprd_var() for i in range(self.N)]

                nb_win_i_0 = [model.add_var() for i in range(self.N)]
                # w_1[i][j] vaut 1 le bidder i gagner le user j et user j est de demographic 1
                w_1 = [[model.add_var(var_type=BINARY) for j in range(self.M)] for i in range(self.N)]
                for j in range(self.M):
                    for i in range(self.N):
                        model += w_1[i][j] == w[i][j] * self.users[j]

                for i in range(self.N):
                    model += l_1 * xsum(w[i][j] for j in range(self.M)) <= xsum(w_1[i][j] for j in range(self.M))

            model.optimize()

            # assert model.num_solutions != 0, 'MIP found no solution'
            if model.num_solutions == 0:
                self.no_solution = True
                return 0

            self.alpha = np.array([alpha[i].x for i in range(self.N)])
            self.prices = np.array([p[j].x for j in range(self.M)])
            self.Q = np.array([[s[i][j].x / p[j].x for j in range(self.M)] for i in range(self.N)])
            self.P = np.array([[s[i][j].x for j in range(self.M)] for i in
                               range(self.N)])  # Les paiements sont splits parmi les winners

        if type(self.fairness_constraint) in [Budget_proportion_constraint]:

            values = {}
            M_gamma = {}
            v_bar = {}
            d = {}
            w = {}
            r = {}
            y = {}
            p = {}
            h = {}
            s = {}

            B = {}  # B DOIT ETRE UNE VARIABLE DU PB ? J'ai bien peu que oui sinon
            # B = [self.bidders.B[i]*eta[gamma][i] for i in range(self.N) for gamma in range self.Gamma] ??

            alpha = {}
            eta = {}

            for gamma in range(self.Gamma):

                values[gamma] = mask(self.bidders.values, self.users, gamma)
                # print(f'values{gamma}',values[gamma])

                M_gamma[gamma] = len(values[gamma][0])

                v_bar[gamma] = [np.max(self.bidders.values[:, j]) for j in range(M_gamma[gamma])]
                d[gamma] = [[model.add_var(var_type=BINARY) for j in range(M_gamma[gamma])] for i in range(self.N)]
                w[gamma] = [[model.add_var(var_type=BINARY) for j in range(M_gamma[gamma])] for i in range(self.N)]
                r[gamma] = [[model.add_var(var_type=BINARY) for j in range(M_gamma[gamma])] for i in range(self.N)]
                y[gamma] = [model.add_var(var_type=BINARY) for i in range(self.N)]

                alpha[gamma] = [model.add_var(lb=0, ub=1) for i in range(self.N)]
                eta[gamma] = [model.add_var(lb=0, ub=1) for i in range(self.N)]

                p[gamma] = [model.add_var() for j in range(M_gamma[gamma])]
                h[gamma] = [model.add_var() for j in range(M_gamma[gamma])]
                s[gamma] = [[model.add_var() for j in range(M_gamma[gamma])] for i in range(self.N)]
                B[gamma] = [model.add_var() for i in range(self.N)]

                # CONSTRAINTS

                # 1
                for i in range(self.N):
                    model += xsum(s[gamma][i][j] for j in range(M_gamma[gamma])) <= B[gamma][i]  # self.bidders.B[i]

                # 2
                for i in range(self.N):
                    # model += xsum(s[gamma][i][j] for j in range(M_gamma[gamma])) >= y[gamma][i] * B[gamma][i]  #self.bidders.B[i]
                    # model += xsum(s[gamma][i][j] for j in range(M_gamma[gamma]))- B[gamma][i] + EPSILON <= y[gamma][i]   #self.bidders.B[i]
                    model += B[gamma][i] - xsum(s[gamma][i][j] for j in range(M_gamma[gamma])) <= (1 - y[gamma][i]) * \
                             self.bidders.B[i]  # self.bidders.B[i]
                # 3
                for i in range(self.N):
                    model += alpha[gamma][i] >= 1 - y[gamma][i]
                # 4
                for j in range(M_gamma[gamma]):
                    model += xsum(s[gamma][i][j] for i in range(self.N)) == p[gamma][j]
                # 5
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += s[gamma][i][j] <= self.bidders.B[i] * d[gamma][i][j]  # self.bidders.B[i]
                # 6
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += h[gamma][j] >= alpha[gamma][i] * values[gamma][i][j]
                # 7
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += h[gamma][j] <= alpha[gamma][i] * values[gamma][i][j] + (1 - d[gamma][i][j]) * \
                                 v_bar[gamma][j]
                # 8
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += w[gamma][i][j] <= d[gamma][i][j]
                # 9
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += p[gamma][j] >= alpha[gamma][i] * values[gamma][i][j] - w[gamma][i][j] * \
                                 values[gamma][i][j]
                # 10
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += p[gamma][j] <= alpha[gamma][i] * values[gamma][i][j] + (1 - r[gamma][i][j]) * \
                                 v_bar[gamma][j]
                # 11
                for j in range(M_gamma[gamma]):
                    model += xsum(w[gamma][i][j] for i in range(self.N)) == 1
                # 12
                for j in range(M_gamma[gamma]):
                    model += xsum(r[gamma][i][j] for i in range(self.N)) == 1
                # 13
                for j in range(M_gamma[gamma]):
                    for i in range(self.N):
                        model += r[gamma][i][j] + w[gamma][i][j] <= 1

                # 14 Fairness constraint:
                if self.fairness_constraint.lam is not None:
                    for i in range(self.N):
                        model += eta[gamma][i] >= self.fairness_constraint.lam * M_gamma[gamma] / self.M
                        # model +=  eta[gamma][i] <= self.fairness_constraint.lam*1/M_gamma[gamma]

                # 15 Constraint on the split of the budget : Definition of the eta
                for i in range(self.N):
                    model += B[gamma][i] == self.bidders.B[i] * eta[gamma][i]
            # 16
            for i in range(self.N):
                model += xsum(eta[gamma][i] for gamma in range(self.Gamma)) == 1

            model.objective = maximize(xsum(p[gamma][j] for gamma in range(self.Gamma) for j in range(M_gamma[gamma])))

            model.optimize()

            # assert model.num_solutions != 0, 'MIP found no solution'
            if model.num_solutions == 0:
                self.no_solution = True
                print('NO SOLUTION')
                return 0

            self.alpha = np.array([[alpha[gamma][i].x for i in range(self.N)] for gamma in range(self.Gamma)])

            self.B_gamma = {gamma: np.array([B[gamma][i].x for i in range(self.N)]) for gamma in range(self.Gamma)}
            self.p_gamma = {gamma: np.array([p[gamma][j].x for j in range(M_gamma[gamma])]) for gamma in
                            range(self.Gamma)}
            self.Q_gamma = {
                gamma: np.array(
                    [[s[gamma][i][j].x / p[gamma][j].x for j in range(M_gamma[gamma])] for i in range(self.N)])
                for gamma in range(self.Gamma)}
            self.eta_gamma = {gamma: np.array([eta[gamma][i].x for i in range(self.N)]) for gamma in range(self.Gamma)}
            self.y_gamma = {gamma: np.array([y[gamma][i].x for i in range(self.N)]) for gamma in range(self.Gamma)}
            self.Q = np.concatenate([self.Q_gamma[gamma] for gamma in range(self.Gamma)], axis=1)

            self.prices = [p[gamma][j].x for gamma in range(self.Gamma) for j in range(M_gamma[gamma])]
            #             self.prices = []

            #             for gamma in range(self.Gamma):
            #                 for j in range(M_gamma[gamma]):
            #                     self.prices += [p[gamma][j].x]
            # print('asdf',self.prices)
            self.P = np.array([[self.Q[i, j] * self.prices[j]] for j in range(self.M) for i in range(self.N)])

    #
    #           self.Q = np.array([[s[i][j].x/p[j].x for j in range(self.M)] for i in range(self.N)])
    #           self.P = np.array([[s[i][j].x for j in range(self.M)] for i in range(self.N)]) # Les paiements sont splits parmi les winners

    def compute_Q(self):
        pass

    def compute_P(self):
        pass

    def report(self):
        super().report()
        print('pacing multipliers \n', self.alpha)
        if type(self.fairness_constraint) in [Budget_proportion_constraint]:
            print(f'B_gamma {self.B_gamma}\n')
            print(f'self.p_gamma {self.p_gamma}\n')
            # print(f'self.y_gamma {self.y_gamma}\n')
            print(f'self.eta_gamma {self.eta_gamma}\n')

    def stats(self):
        super().stats()


class AdaptivePacing(Mechanism):
    """
    Adaptive pacing mechanism. We consider that every player is using an adaptive pacing mechanism.
    From 'Balseiro 2019, Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium'
    """

    def __init__(self, bidders, users, fairness_constraint, ):
        super().__init__(bidders, users, fairness_constraint)

        self.mu = np.zeros((self.N, self.M))
        # here we can init mu[:,0] with a custom values!
        self.rho = np.array(self.bidders.B) / self.M  # target expenditure
        self.eps = [(self.M) ** (-0.5)] * self.N

        self.utility = None
        self.prices = np.zeros(self.M)
        self.bids = np.zeros((self.N, self.M))  # for truthfull mechanism the bids equals self.bidders.values, not here
        self.Budgets = np.zeros((self.N, self.M))
        self.Budgets[:, 0] = self.bidders.B

        # assert isinstance(bidders,Bidders_budget),"Multiplicative Pacing mechanism needs budget" #Ne marche pas actuellement voir comment check des sous classes!
        self.solve()
        self.compute_Q()
        self.compute_P()
        self.stats()
        self.no_solution = False

    def compute_best_price(self):
        return np.amax(self.bids, axis=0)

    def compute_second_best_price(self):
        return [np.sort(np.unique(self.bids[:, j]))[-2] for j in
                range(self.bidders.M)]  # pb si il tous les bidders mettemt le meme prix!

    def solve(self):
        mu_bar = np.ones(self.N) * 10
        for j in range(self.M):
            for i in range(self.N):
                self.bids[i, j] = min(self.bidders.values[i, j] / (1 + self.mu[i, j]), self.Budgets[i, j])

            best_price = np.amax(self.bids[:, j])


            second_best_price = np.sort(self.bids[:, j])[-2]
            self.prices[j] = second_best_price
            winners = np.array(np.argwhere(self.bids[:, j] == best_price).flatten().tolist())
            self.Q[winners, j] = 1 / len(winners)
            winner = np.random.choice(range(self.bidders.N), p=self.Q[:, j])

            self.P[winner, j] = second_best_price
            if j < self.M - 1:
                self.Budgets[:, j + 1] = self.Budgets[:, j]
                self.Budgets[winner, j + 1] -= second_best_price
                for i in range(self.N):
                    self.mu[i, j + 1] = min(max(self.mu[i, j] - self.eps[i] * (self.rho[i] - self.P[i, j]), 0),
                                            mu_bar[i])

    def compute_Q(self):
        pass

    def compute_P(self):
        pass

    def report(self):
        super().report()
        print('Budgets evolution \n', self.Budgets)
        print('Mu \n', self.mu)



class IndividualAdaptivePacing(Mechanism):
    """
    Adaptive pacing mechanism. We consider that every player is using an adaptive pacing mechanism.
    From 'Balseiro 2019, Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium'
    """

    def __init__(self, bidders, users, fairness_constraint, ):
        super().__init__(bidders, users, fairness_constraint)

        self.mu = np.zeros((self.N, self.M))
        # here we can init mu[:,0] with a custom values!
        self.rho = np.array(self.bidders.B) / self.M  # target expenditure
        self.eps = [(self.M) ** (-0.5)] * self.N

        self.utility = None
        self.prices = np.zeros(self.M)
        self.bids = np.zeros((self.N, self.M))  # for truthfull mechanism the bids equals self.bidders.values, not here
        self.Budgets = np.zeros((self.N, self.M))
        self.Budgets[:, 0] = self.bidders.B

        # assert isinstance(bidders,Bidders_budget),"Multiplicative Pacing mechanism needs budget" #Ne marche pas actuellement voir comment check des sous classes!
        self.solve()
        self.compute_Q()
        self.compute_P()
        self.stats()
        self.no_solution = False

    def compute_best_price(self):
        return np.amax(self.bids, axis=0)

    def compute_second_best_price(self):
        return [np.sort(np.unique(self.bids[:, j]))[-2] for j in
                range(self.bidders.M)]  # pb si il tous les bidders mettemt le meme prix!

    def solve(self):
        mu_bar = np.ones(self.N) * 10
        for j in range(self.M):

            self.bids[0, j] = min(self.bidders.values[0, j] / (1 + self.mu[0, j]), self.Budgets[0, j])

            best_price = np.amax(self.bids[:, j])


            second_best_price = np.sort(self.bids[:, j])[-2]
            self.prices[j] = second_best_price
            winners = np.array(np.argwhere(self.bids[:, j] == best_price).flatten().tolist())
            self.Q[winners, j] = 1 / len(winners)
            winner = np.random.choice(range(self.bidders.N), p=self.Q[:, j])

            self.P[winner, j] = second_best_price
            if j < self.M - 1:
                self.Budgets[:, j + 1] = self.Budgets[:, j]
                self.Budgets[winner, j + 1] -= second_best_price
                for i in range(self.N):
                    self.mu[i, j + 1] = min(max(self.mu[i, j] - self.eps[i] * (self.rho[i] - self.P[i, j]), 0),
                                            mu_bar[i])

    def compute_Q(self):
        pass

    def compute_P(self):
        pass

    def report(self):
        super().report()
        print('Budgets evolution \n', self.Budgets)
        print('Mu \n', self.mu)




class Hindsight(Mechanism):
    """
    Given sequences of realized valuations v_k and highest competing bids d_k, we denote by π_H the highest
    performance achieved with the benefit of hindsight for user k. The strategy of players other than k is truthfull bidding and the mechanism is second price auction.
    From 'Balseiro 2019, Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium'
    """

    # bids of all users except k, k
    def __init__(self, bidders, users, fairness_constraint):
        super().__init__(bidders, users, fairness_constraint)

        self.utility = None

        self.bids = self.bidders.values  # for truthfull mechanism the bids equals self.bidders.values, not here

        self.k = 0  # CHANGE HERE THE DEFAULT VALUE
        # assert isinstance(bidders,Bidders_budget),"Multiplicative Pacing mechanism needs budget" #Ne marche pas actuellement voir comment check des sous classes!
        self.solve()
        self.compute_Q()
        self.compute_P()
        self.stats()
        self.no_solution = False

    def solve(self):
        competing_bids = np.delete(self.bids, self.k, axis=0)

        d_k = [np.sort(competing_bids[:, j])[-1] for j in range(self.bidders.M)]

        v_k = self.bids[self.k, :]
        utility_k = [v_k[j] - d_k[j] for j in range(self.bidders.M)]

        model = Model("knapsack")
        model.verbose = 0

        x = [model.add_var(var_type=BINARY) for j in range(self.bidders.M)]
        model.objective = maximize(xsum(utility_k[i] * x[i] for i in range(self.bidders.M)))
        model += xsum(d_k[i] * x[i] for i in range(self.bidders.M)) <= self.bidders.B[self.k]

        model.optimize()
        selected = [i for i in range(self.bidders.M) if x[i].x >= 0.99]

        # We now run a classical Vickerey Auction
        self.bids = np.copy(self.bidders.values)
        self.bids[self.k, :] = 0
        for el in selected:
            self.bids[self.k, el] = self.bidders.values[self.k, el] # advertiser k only bids on object he want to win
        self.best_price = np.amax(self.bids, axis=0)
        self.second_best_price = [np.sort(self.bids[:, j])[-2] for j in range(self.bidders.M)]
        self.prices = self.second_best_price

    def compute_Q(self):

        for j in range(self.bidders.M):
            winners = np.array(np.argwhere(self.bids[:, j] == self.best_price[j]).flatten().tolist())
            self.Q[winners, j] = 1 / len(winners)

    def compute_P(self):
        for j in range(self.bidders.M):
            winner = np.random.choice(range(self.bidders.N), p=self.Q[:, j])
            self.P[winner, j] = self.second_best_price[j]

    def report(self):
        super().report()
