import numpy as np
import pandas as pd
from scipy.stats import kruskal, f_oneway 



def correlation_ratio(categories, measurements):
    '''Gives the correlation ratio between a categorical variable and a numerical one
    '''
    ## Numerically coding the categorical variable: assigning an integer (starting from 0) to the categories at hand
    fcat, _ = pd.factorize(categories)

    ## number of categories
    cat_num = np.max(fcat)+1

    ## Getting the average of the numerical value per category
    y_avg_array = np.zeros(cat_num)

    ## Getting the number of instances per category (the population of each category)
    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):
        ## we get the values of the numerical variables for each individual of category i
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = measurements.mean()
    numerator = np.sum(n_array@(y_avg_array - y_total_avg)**2)
    #numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum((measurements - y_total_avg)**2 )

    return numerator/denominator




def association_test(categories, test_level=0.05, test_type="ANOVA", printing=True):
    '''
    Tests the association between a numerical N and categorical C variabes, the values of N per categories are in categories array       like
    '''
    if test_type=='Kruskal Wallis':
        pval_kruskal=kruskal(*categories)[1]
        if printing:
            if pval_kruskal <= test_level:
                return("The Kruskal Wallis test rejects the equality of medians at level "+str(test_level))
            return ("The Kruskal Wallis test rejects the equality of medians at level"+str(test_level))
        else: return pval_kruskal
    
    elif test_type=='ANOVA':
        pval_anova=f_oneway(*categories)[1]
        if printing:
            if pval_anova <= test_level:
                return("The ANOVA test rejects the equality of means at level "+str(test_level))
            return ("The Kruskal Wallis test rejects the equality of means at level"+str(test_level))
        return pval_anova