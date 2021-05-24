import numpy as np

class Auction():
    def __init__(self, bidders, mechanism, users=None, fairness_constraint=None):
        '''
        bidders: List of objects of type bidders
        mechanism: the mecanism of the auction
        M: number objects
        '''
        self.N = bidders.N
        self.M = bidders.M
        self.bidders = bidders
        if users is not None:
            assert len(users) == self.M, 'Demographics type of users must match the number of users M'
        self.users = np.array(users)
        self.fairness_constraint = fairness_constraint
        self.mechanism = mechanism(bidders, np.array(self.users), fairness_constraint)

    def run_auction(self):
        if self.bidders.values is None:
            self.bidders.sample_values()
        self.mechanism(self.bidders)

    def report(self):
        self.mechanism.report()

    def multiple_run(self, n):
        pass

    def plot(self):
        """Plot the result of the auction (utility, welfare)"""
        pass