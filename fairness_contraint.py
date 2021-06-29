class Fairness_constraint():
    def __init__():
        pass


class Proportion_constraint(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam

class Proportion_constraint_cumulative(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam


class Budget_proportion_constraint(Fairness_constraint):
    def __init__(self, lam=None, proportion=None):
        self.lam = lam
        self.proportion = None
