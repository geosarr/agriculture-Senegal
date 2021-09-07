from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from pandas import DataFrame
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric import kernel_regression
from sklearn.model_selection import KFold
          
metrics={}
          
def display(scores):
    print("Scores", scores)
    print("Average score", scores.mean())
    print("Standard deviation of scores", scores.std())
          


def plot_learning_curve(model, X, Y, val_ratio = 0.25, nb_training = 100):
    '''Plot the learning curve, model is a model pretrained, X is the dataframe of the explaining variables ,
    Y is the target, val_ratio is the ratio of data for evaluation
    nb_training designates the number of mini batches
    '''
    ## One can fix the seed
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = val_ratio)
    train_errors, val_errors =  [], []
    #print(X_train)
    for m in range(1, len(X_train), len(X_train)//nb_training):
        model.fit(X_train[:m], Y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_val_pred =  model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_pred, Y_train[:m]))
        val_errors.append(mean_squared_error(y_val_pred, Y_val))
    plt.figure(figsize = (20,10))
    plt.plot(np.sqrt(train_errors), label = 'RMSE on training set')
    plt.plot(np.sqrt(val_errors), label = 'RMSE on validation set')
    plt.plot([Y.mean()]*len(train_errors), label = 'Mean of the target')
    plt.xlabel('$n^{th}$ batch over ' + str(nb_training) + ' batches')
    plt.ylabel('Error')
    plt.legend()

    
    
def rmse(Y, Y_pred):
    '''
    Computes the Root Mean Squared Error Metrics
    '''
    return sqrt(mean_squared_error(Y,Y_pred))
    
    
    
def kfold(model, X, Y, k):
    '''
    Cross Validation (k fold)
    '''
    scores=- cross_val_score(model, X, Y, scoring = 'neg_mean_squared_error', cv = k)
    display(sqrt(scores))



def test_models(models, test_X, test_Y):
    '''Measures the performance of the models on the test set for comparison,
    Here the performance measure is the RMSE
    '''
    d = DataFrame(columns=["Features", "Errors"])
    for model in models:
        d = d.append(DataFrame([[type(model).__name__, np.sqrt(mean_squared_error(model.predict(test_X), test_Y))]],\
                                  columns=["Features", "Errors"]))
    d=d.sort_values("Errors")
    d.reset_index(inplace=True)
    return d



def cv_np_reg(X, Y, k=5):
    '''
    Cross validation for np reg model with k folds, all the regressors must be continuous
    Output the RMSE of the 
    '''
    ## Creating the folds
    kf = KFold(n_splits=k, shuffle=True)
    L = kf.split(X)
    evaluation=[0]*k
    for pos, indices in enumerate(L):
        train_indices, test_indices = indices
        X_train, X_test = X.iloc[train_indices,:], X.iloc[test_indices, :]
        Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
        np_reg  = kernel_regression.KernelReg(endog = Y_train, exog = X_train, var_type = 'c'*X.shape[1])
        evaluation[pos] = np.sqrt(mean_squared_error(np_reg.fit(X_test)[0], Y_test
                                                    )
                                 )
        print(str(pos+1)+ " validation(s) made")
    display(np.array(evaluation))