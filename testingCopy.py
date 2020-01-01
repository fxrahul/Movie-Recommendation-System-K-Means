# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:47:55 2019

@author: Rahul
"""
#import pandas as pd
#a = pd.DataFrame({"A":[0,1,2],"E":[0,1,2],"C":[4,5,6]})
#b = pd.DataFrame({"D":[0,1,2,5,6],"E":[0,1,2,3,6],"F":[4,5,6,7,8]})
#dfinal = a.merge(b, how='inner', left_on='A', right_on='D')
#print(dfinal)
#a["C"] = dfinal["E_y"]
#print(a)
##
#from sklearn.cluster import KMeans
#import numpy as np
#X = np.array([[1, 2], [1, 4], [1, 0],
#              [10, 2], [10, 4], [10, 0]])
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#kmeanswithiness = KMeans(n_clusters=2, random_state=0).fit_transform(X)
#
#print(kmeans.cluster_centers_)

#import numpy as np
#a = np.zeros(3)
#print(a)

import pandas as pd
df = pd.DataFrame({'itemid': [1,2,3,4,5],
                   'rating': [2,3.5,3,4.5,4],
                   'cluster': [1,2,2,2,2]})

#print(df["cluster"][2])
like = df.groupby(["cluster"]).mean().drop(labels=["itemid"],axis=1).reset_index(drop=True)
clusterNo=list()
for i in range(len(like)):
    clusterNo.append(i+1)

like = like.assign(cluster=clusterNo)
#
#maximumRating = max(like["rating"])
#vector = like.loc[like['rating'] == maximumRating]
#like = vector.iloc[0][1].astype(int) 
print(like)

