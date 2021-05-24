import numpy as np

class Bidders():
    def __init__(self, X_NM=None, values=None):
        '''
        N: number of bidders
        M: number of objects/users
        X_NM: a matrix of random variable where X_NM[i][j] has the distribution of the value of user i for object j.

        '''
        self.X_NM = X_NM
        self.values = np.array(values)
        self.N, self.M = (len(X_NM), len(X_NM[0])) if X_NM is not None else (len(values), len(values[0]))

    def sample_values(self):
        assert self.X_NM is not None, 'X_NM is None : To sample the values from bidders, distribution have to be provided.'
        self.values = np.array([[self.X_NM[i][j].sample() for j in range(self.M)] for i in range(self.N)])


class Bidders_budget(Bidders):
    """
    Bidder with extra attribute B for bids
    """

    def __init__(self, B, X_NM=None, values=None):
        super().__init__(X_NM, values)
        self.B = B
        assert len(B) == self.N, 'Number of bidders must be equals to the number of bids'
