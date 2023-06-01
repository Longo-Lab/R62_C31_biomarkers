#!/usr/bin/env python

import re
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ML groups
group = ['svm_knn', 'svm_rfc', 'xgb_knn', 'xgb_rfc']
analysis = 'drug'

# regex to fetch values
p = re.compile("top\s(\d+)\sf"
               ".*?Top.*?ed\):\s(\S+)\n"
               ".*?curacy\):\s(\S+)\n"
               ".*?hted\savg\s+(\S+)\s+(\S+)\s"
               ".*?orted\):\s(\S+)", re.S)

# add to df
df = pd.DataFrame(columns=['group', 'file', 'n_features', 'top_features',
                           'accuracy', 'precision', 'recall', 'unused_features'])
for g in group:
    for i in range(1000):
        with open(g + str(i) + ".txt", 'r') as file:
            m = p.findall(file.read())
            for x in m:
                df.loc[len(df.index)] = [g, i, *x]

# convert columns to numeric
for x in ['file', 'n_features', 'accuracy', 'precision', 'recall']:
    df[x] = pd.to_numeric(df[x])

# write to file
df.to_csv('ML.' + analysis + '.outcomes.csv')

## Plot
# Initialize the figure
f, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(16,16))
sns.despine()

# Show the conditional means
sns.pointplot(x="n_features", y="accuracy", hue="group",
              data=df, dodge=False, palette="Paired",
              capsize=0.2 , ci="sd", errwidth=1, ax=ax1)
ax1.set_title("Prediction Accuracy", fontsize = 25)

# Show the conditional means
sns.pointplot(x="n_features", y="precision", hue="group",
              data=df, dodge=False, palette="Paired",
              capsize=0.2 , ci="sd", errwidth=1, ax=ax2)
ax2.set_title("Prediction Precision", fontsize = 25)

# Show the conditional means
sns.pointplot(x="n_features", y="recall", hue="group",
              data=df, dodge=False, palette="Paired",
              capsize=0.2 , ci="sd", errwidth=1, ax=ax3)
ax3.set_title("Prediction Recall", fontsize = 25)

# clean up and print
plt.tight_layout()
plt.savefig('ML.' + analysis + '.outcomes.pdf')
