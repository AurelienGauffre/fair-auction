class Fairness_constraint():
    def __init__():
        pass
class Linear_constraint(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam

class Linear_constraint_absoAP(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam
class Linear_constraint_absoAPC(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam

class Linear_constraint_ratioAP(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam
class Linear_constraint_ratioAPC(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam

class Linear_constraint_cumulative(Fairness_constraint):
    def __init__(self, lam):
        self.lam = lam



class Budget_proportion_constraint(Fairness_constraint):
    def __init__(self, lam=None, proportion=None):
        self.lam = lam
        self.proportion = None
