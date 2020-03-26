import numpy as np

class TaskGenerator:
    def __init__(self, d):
        
        self.d = d
        u, s, vt = np.linalg.svd(np.random.randn(self.d,self.d))
        s[1] = s[0]/2
        s[2:] = 0
        self.omega = u @ np.diag(s) @ vt

    def sample_task(self):

        alpha = np.random.randn(self.d)

        w_star = self.omega@alpha 

        return w_star + 1

    
class DataGenerator:
    def __init__(self, weight, d):
        self.weight = weight
        self.d = d
        
    def sample_data(self, N, X = None):
        if X is None:
            X = np.hstack((np.ones((N, 1)), np.random.randn(N, self.d)))
        y = X @ self.weight + np.random.randn(N)
        return X, y
    
    def sample_train(self, N, X_tr = None):
        if X_tr is None:
            X_ = np.hstack((np.ones((4*N, 1)), np.random.randn(4*N, self.d)))
            X_subset = X_[np.where(X_[:,-self.d] < 0)[0],:]
            X_tr = X_subset[:N,:]
        y_tr = X_tr @ self.weight + np.random.randn(N)
        return X_tr, y_tr
    
    def sample_test(self, N, X_ts = None):
        if X_ts is None:
            X_ = np.hstack((np.ones((4*N, 1)), np.random.randn(4*N, self.d)))
            X_subset = X_[np.where(X_[:,-self.d] > 0)[0],:]
            X_ts = X_subset[:N,:]

        y_ts = X_ts @ self.weight + np.random.randn(N)
        return X_ts, y_ts    
        
        
        