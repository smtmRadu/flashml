import numpy as np

#https://github.com/udacity/deep-reinforcement-learning/blob/master/
#dxt=θ(μ−xt)dt+σdWt
class OrnsteinUhlenbeckProcess:
    '''
    Generates OU noise.
    '''
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dtype=np.float32):
        self.xt = np.zeros(size, dtype=dtype)
        self.mu = mu * np.ones(size, dtype=dtype)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        self.dtype =dtype

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.xt = np.zeros_like(self.xt)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.xt
        dxt = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.xt.shape)
        self.xt += dxt
        return self.xt