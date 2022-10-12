from ensurepip import version
from pickle import TRUE
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
nalphabets = len(alphabets)

samples_per_alphabets = 5
figure = plt.figure(figsize=(nalphabets*2,(1+samples_per_alphabets*2)))

idx_cls = 0
for cls in alphabets:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samples_per_alphabets, replace=False)
  i = 0
  for idx in idxs:
    plt_idx = i * nalphabets + idx_cls + 1
    p = plt.subplot(samples_per_alphabets, nalphabets, plt_idx);
    p = sns.heatmap(np.array(X.loc[idx]).reshape(28,28), cmap=plt.cm.gray, 
             xticklabels=False, yticklabels=False, cbar=False);
    p = plt.axis('off');
    i += 1
  idx_cls += 1

idxs = np.flatnonzero(y == '0')
print(np.array(X.loc[idxs[0]]))


print(len(X))
print(len(X.loc[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0


print(X.loc[0])
print(y.loc[0])


