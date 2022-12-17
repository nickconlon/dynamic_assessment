import numpy as np


class q_policy:
    """
    Class to encapsulate a single policy
    """
    def __init__(self, policy):
        self.q_table = np.load(policy)

    def pi(self, state):
        return np.argmax(self.q_table[state])

    def noisy_pi(self, state, p_correct):
        a = np.argmax(self.q_table[state])
        actions = np.arange(4)
        ps = np.zeros_like(actions) + (1-p_correct)/(len(actions)-1)
        ps[a] = p_correct
        a = np.random.choice(a=actions, p=ps)
        return a


class q_policies:
    """
    Class to encapsulate a set of policies
    """
    def __init__(self, policies, targets):
        self.policies = {}
        for policy, target in zip(policies, targets):
            self.policies[target] = q_policy(policy)

    def get_policy(self, state, target, p_correct=None):
        if p_correct is not None:
            return self.policies[target].noisy_pi(state, p_correct)
        else:
            return self.policies[target].pi(state)
