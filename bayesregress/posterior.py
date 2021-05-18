
class LogPosterior(object):
    def __init__(self, log_prior, log_likelihood):
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood

    def __call__(self, params):
        return self.log_prior(params) + self.log_likelihood(params)
