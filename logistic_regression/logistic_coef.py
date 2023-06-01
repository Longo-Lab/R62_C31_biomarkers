#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def get_log(term, titl):
    '''
    Loops through and graphs log reg importance
    term = list term for _y and _X prefixes, also outname
    saves file returns nothing
    '''
    y = globals()[term + '_y']
    X = globals()[term + '_X']
    model = LogisticRegression()
    model.fit(X, y)
    imp = pd.DataFrame(zip(model.coef_[0],geno_X.columns),columns=['coef', 'feats'])
    
    fig, ax = plt.subplots(figsize=(16,4))
    sns.barplot(x='feats', y='coef', data=imp)
    plt.xticks(rotation=45)
    ax.set_title('Z-scaled Logistic Coefficients for {}'.format(titl))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Coefficient')
    show_values_on_bars(ax)
    plt.tight_layout()
    plt.savefig('{}.logistic_coef.pdf'.format(term))
    return

if __name__ == "__main__":
    df = pd.read_csv('knn_input.csv')
    geno_y = df[df['TX'] < 3]['TX'].map({1:0, 2:1})
    geno_X = df[df['TX'] < 3].drop('TX', axis=1)
    drug_y = df[df['TX'] > 1]['TX'].map({2:0, 3:1})
    drug_X = df[df['TX'] > 1].drop('TX', axis=1)
    recovery_y = df['TX'].replace(3, 1).map({1:0, 2:1})
    recovery_X = df.drop('TX', axis=1)
    lis={'geno':'Genotype', 'drug':'Drug Effect', 'recovery':'WT+Treated vs Tg'}
    [get_log(x,y) for x,y in lis.items()]

