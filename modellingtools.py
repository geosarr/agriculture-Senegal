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

          
metrics={"RMSE": "neg_mean_squared_error", "MAE": "neg_mean_absolute_error", "MedAE": "neg_median_absolute_error"}


          
def display(scores, score_name, width=0.3, ax=None, xlabel="", ylabel="", title=""):
    x = range(1, len(scores)+1)
    if ax==None:
        plt.bar(x, scores, width, label=score_name, color="green")
        plt.plot(x, [scores.mean()]*len(scores), label="mean "+score_name)
        plt.plot(range(1, len(scores)+1), [scores.std()]*len(scores), label="standard deviation "+score_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
    else:
        ax.bar(x, scores, width, label=score_name, color="green")
        ax.plot(x, [scores.mean()]*len(scores), label="mean "+score_name)
        ax.plot(range(1, len(scores)+1), [scores.std()]*len(scores), label="standard deviation "+score_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

          


def plot_learning_curve(model, X, Y, val_ratio = 0.25, nb_training = 100, rand_state=20, put_mean_target=True):
    '''Plot the learning curve, model is a model pretrained, X is the dataframe of the explaining variables ,
    Y is the target, val_ratio is the ratio of data for evaluation
    nb_training designates the number of mini batches
    '''
    #set.
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = val_ratio, random_state=rand_state)
    train_errors, val_errors =  [], []
    #print(X_train)
    for m in range(1, len(X_train), len(X_train)//nb_training):
        model.fit(X_train[:m], Y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_val_pred =  model.predict(X_val)
        train_errors.append(sqrt(mean_squared_error(y_train_pred, Y_train[:m])))
        val_errors.append(sqrt(mean_squared_error(y_val_pred, Y_val)))
    plt.figure(figsize = (20,10))
    plt.plot(train_errors, label = 'RMSE on training set')
    plt.plot(val_errors, label = 'RMSE on validation set')
    if put_mean_target:
        plt.plot([Y.mean()]*len(train_errors), label = 'Mean of the target')
    plt.xlabel('$n^{th}$ batch over ' + str(nb_training) + ' batches')
    plt.ylabel('Error')
    plt.legend()

    
    
def rmse(Y, Y_pred):
    '''
    Computes the Root Mean Squared Error Metrics
    '''
    return sqrt(mean_squared_error(Y,Y_pred))
    
    
    
def kfold(model, X, Y, k, metrics_name="RMSE", ax=None, xlabel="", ylabel="", title=""):
    '''
    Cross Validation (k fold)
    '''
    scores=- cross_val_score(model, X, Y, scoring = metrics[metrics_name], cv = k)
    if metrics_name=="RMSE": display(sqrt(scores), score_name=metrics_name, ax=ax)
    else: display(scores, score_name=metrics_name, ax=ax, xlabel=xlabel, ylabel=ylabel, title=title)



def test_models(models, test_X, test_Y):
    '''Measures the performance of the models on the test set for comparison,
    Here the performance measure is the RMSE
    '''
    d = DataFrame(columns=["Model", "RMSE"])
    for model in models:
        temp=DataFrame([[type(model).__name__, rmse(model.predict(test_X), test_Y)]],\
                                  columns=["Model", "RMSE"])
        d = d.append(DataFrame([[type(model).__name__, rmse(model.predict(test_X), test_Y)]],\
                                  columns=["Model", "RMSE"]))
    d=d.sort_values("RMSE")
    d.reset_index(inplace=True, drop=True)
    return d



def cv_np_reg(X, Y, k=5):
    '''
    Cross validation for non parametric regression model with k folds, all the regressors must be continuous
    Output the RMSE of the predictions
    '''
    ## Creating the folds
    kf = KFold(n_splits=k, shuffle=True)
    L = kf.split(X)
    evaluation=[0]*k
    for pos, indices in enumerate(L):
        train_indices, test_indices = indices
        X_train, X_test = X[train_indices,:], X[test_indices, :]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        np_reg  = kernel_regression.KernelReg(endog = Y_train, exog = X_train, var_type = 'c'*X.shape[1])
        evaluation[pos] = np.sqrt(mean_squared_error(np_reg.fit(X_test)[0], Y_test
                                                    )
                                 )
        print(str(pos+1)+ " validation(s) made")
    display(np.array(evaluation), "KernelReg")