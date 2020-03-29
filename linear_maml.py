import numpy as np


class LinearMetaLearn:
    def __init__(self, X_trains, X_tests, y_trains, y_tests, alpha, num_outer_loop_epochs, 
                 num_inner_loop_epochs = 1, second_order = True, w = None):
        self.X_trains = X_trains
        self.X_tests = X_tests
        self.y_trains = y_trains
        self.y_tests = y_tests
        self.alpha = alpha
        self.second_order = second_order
        self.num_outer_loop_epochs = num_outer_loop_epochs 
        self.num_inner_loop_epochs = num_inner_loop_epochs 
        if w is None:
            self.w = np.zeros(2)
        else:
            self.w = w
            
        self.mses = []
        
    def inner_loop_updates(self, X, y, num_epochs):
        for _ in range(num_epochs):
            y_pred = X @ self.w
            delta = (y_pred - y)[:,np.newaxis]
            self.w = self.w - self.alpha * np.sum(delta * X,0) 
    
    def predict(self, X):
        return X @ self.w
    
    
    def iterate(self):
        for outer_loop_indx in range(self.num_outer_loop_epochs):
            for (X_tr, y_tr, X_ts, y_ts) in zip(self.X_trains, self.y_trains, 
                                                self.X_tests, self.y_tests):
                
                self.inner_loop_updates(X_tr, y_tr, 
                                        self.num_inner_loop_epochs)
                y_pred_ts = X_ts @ self.w
                delta = (y_pred_ts - y_ts)[:,np.newaxis]
                if self.second_order:
                    self.w = self.w - self.alpha *np.sum((delta * X_ts) @ 
                                                         (np.eye(2) - 
                                                          self.alpha * X_tr.T @ X_tr), 0)
                else:
                    self.w = self.w - self.alpha *np.sum(delta * X_ts,0)
                                        
                delta_tr = (X_tr @ self.w - y_tr)**2
                delta_ts = (X_ts @ self.w - y_ts)**2

                self.mses.append(np.mean(np.hstack((delta_tr, delta_ts))))
            
