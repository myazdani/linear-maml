import numpy as np

class OverfitMetaLearn:
    def __init__(self, X_tr, X_ts, y_tr, y_ts, alpha, second_order, 
                 num_outer_loop_epochs, num_inner_loop_epochs = 1, w = None):
        '''
        Args:
        -----
        X_tr : numpy 2d array
            training input data with rows corresponding to different examples
        y_tr : numpy 1d array
            training target values
        X_ts : numpy 2d array
            test input data with rows corresponding to different examples
        y_ts : numpy 1d array
            test target values
        alpha : float 
            learning rate
        second_order : numpy array
            Square matrix used for outer-loop update. Should be d-by-d where
            d is the number of features. If unknown use np.eye(d)
        num_outer_loop_epochs : int
            Number of iterations for outer loop
        num_inner_loop_epochs : int (default)
            Number of iterations for inner loop        
        w : numpy array
            1d numpy array corresponding to the coeficients of linear model. 
            
        
        Example:
        --------
        >> w_init = 0*np.random.randn(2)
        >> maml = OverfitMetaLearn(X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, alpha = .01,
                               second_order = (np.eye(2) - alpha * X_tr.T @ X_tr), 
                               num_outer_loop_epochs = 10000, 
                               num_inner_loop_epochs=1, 
                               w = w_init)

        >> foml = OverfitMetaLearn(X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, alpha = .01,
                               second_order = np.eye(2), 
                               num_outer_loop_epochs = 10000, 
                               num_inner_loop_epochs=1, 
                               w = w_init)


        >> woml = OverfitMetaLearn(X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, alpha = .01,
                               second_order = (np.eye(2) - alpha * X_ts.T @ X_ts), 
                               num_outer_loop_epochs = 10000, 
                               num_inner_loop_epochs=1, 
                               w = w_init)        
                               
        >> maml.iterate()
        >> foml.iterate()
        >> woml.iterate()
        '''
        self.X_tr = X_tr
        self.X_ts = X_ts
        self.y_tr = y_tr
        self.y_ts = y_ts
        self.alpha = alpha
        self.second_order = second_order
        self.num_outer_loop_epochs = num_outer_loop_epochs 
        self.num_inner_loop_epochs = num_inner_loop_epochs 
        if w is None:
            self.w = np.zeros(X_tr.shape[1])
        else:
            self.w = w
            
        self.mses = []
    
    
    def iterate(self):
        for outer_loop_indx in range(self.num_outer_loop_epochs):
            for _ in range(self.num_inner_loop_epochs):
                y_pred = self.X_tr @ self.w
                delta = (y_pred - self.y_tr)[:,np.newaxis]
                self.w = self.w - self.alpha * np.sum(delta * self.X_tr,0)

            y_pred_ts = self.X_ts @ self.w
            delta = (y_pred_ts - self.y_ts)[:,np.newaxis]
            self.w = self.w - self.alpha *np.sum((delta * self.X_ts) @ self.second_order, 0)
            
            delta_tr = (self.X_tr @ self.w - self.y_tr)**2
            delta_ts = (self.X_ts @ self.w - self.y_ts)**2
            
            self.mses.append(np.mean(np.hstack((delta_tr, delta_ts))))
            
    def predict(self, X):
        return X @ self.w
            