import gmaps as gmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import pandas as pd



def plot_gmap(villages_location, my_key=""):
    '''
    Using google maps to plot a map on a Jupyter Notebook: look at the documentation of gmaps library to see how to properly
    install it depending on your IDE.
    Warning villages_location should be a dataframe of columns ["site_name", "latitude", "longitude"]
    '''
    # configuring gmaps
    gmap.configure(api_key=my_key)
    
    # building the maps
    fig = gmap.figure(map_type = 'SATELLITE', center = (14.3390167, -16.4111425), zoom_level = 10)
    villages_location = gmap.symbol_layer(villages_location[["latitude", "longitude"]], 
                                         fill_color='black', stroke_color='black', scale=1)
    fig.add_layer(villages_location)
    fig

    
    
def plot_feature_importance(X, Y, model=RandomForestRegressor()):
    '''
    Plotting the feature importances by default using a RandomForestRegression model.
    '''
    model.fit(X, y=Y)

    # get importance
    importance = model.feature_importances_

    # summarize feature importance
    importance = pd.DataFrame({"features": X.columns, "importance": importance}).sort_values("importance")
    
    importance.plot(x='features', y="importance", kind='barh', figsize=(15,8))

    
    
def plot_plotly(positions):
    '''
    Plotting a map using plotly
    Warning positions should be a dataframe of columns ["site_name", "latitude", "longitude"].
    '''
    fig = px.scatter_mapbox(
    positions, lat=positions.columns[1], lon=positions.columns[2], hover_name=positions.columns[0],
    color_discrete_sequence=["fuchsia"], zoom=2, height=300
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    
    
    
def plot_association(data, association_type, ax=None, vmin=-1, vmax=1, masks=True, cmap="YlGnBu"):
    if ax==None: f, ax = plt.subplots(figsize=(10, 6))
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = masks
    sns.heatmap(data, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, \
                annot=True, ax=ax)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45);
    ax.set_title("Heatmap of "+ association_type);
    

    
def box_plot(data, x, y, xlabel="", ylabel=""):
    '''
    '''
    f, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45);
    
    
    
def plot_feature_cross_target(data: pd.DataFrame, target: str, num_feat: list, cat_feat: list, nb_cuts: int=10, n_col: int=2, figsize: tuple=(20,10), fontsize: int=16):
    '''
    Plotting the proportion of data belonging in the inter quantile intervals or groups of the explaining features cross 
    the inter quantile interval of the target feature
    '''
    d=(
        pd.concat(
        [
            data[num_feat].apply(lambda x: pd.qcut(x, q=nb_cuts, duplicates='drop', precision=3), axis=0), 
            data[[target]], data[cat_feat]
        ], axis=1)
      )
    p=len(num_feat)+len(cat_feat)
    n_row=p//n_col+min(1, p%n_col)
    fig, ax=plt.subplots(n_row, n_col , figsize=figsize)
    for i, col in enumerate(num_feat+cat_feat):
        (
         d[[col, target]]
         .groupby(col)
         .mean()
         .reset_index()
         .sort_values(col)
         .plot(kind="barh", x=col, y=target, ax=ax[i//n_col, i%n_col], legend=False)
        )
    if p%n_col!=0:
        for i in range(p, n_col*n_row):
            ax[i//n_col, i%n_col].remove()
    fig.suptitle("Mean of "+target + " cross explaining features", fontsize=fontsize)
    
    
    