# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:30:49 2019

@author: Rahul
"""

import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



data = pd.read_csv('movies.csv')
df = pd.DataFrame(data, columns = ['movieId', 'title', 'genres'])
#print(inputData[0][-1])

#---------------------------------checking number of missing values in each column-----------------------------
#dataset = pd.read_csv('movies.csv', header=None)
#print((dataset[[0,1,2]] == '(no genres listed)').sum())
#0       0
#1       0
#2    4266
#dtype: int64

   #maximun missing values in genres column

#------------------------------------------Identifying and removing missing values------------------------------
df = df.replace('(no genres listed)',np.NaN)
nan = [np.NaN]
df = df[~df['genres'].isin(nan)]
#--------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------


#-----------------------------------converting genres to multi-labels------------------------------------------
genres = []
for i in range(df.shape[0]):
    checkFirstNoDuplicateGenre = (df.iat[i,-1]).split("|")
    for j in range(len(checkFirstNoDuplicateGenre)):
        flag = 1
        for k in range(len(genres)):
            if checkFirstNoDuplicateGenre[j] == genres[k]:
                flag = 0
                break
        if flag == 1:
            genres.append(checkFirstNoDuplicateGenre[j])
            
        
#---------------------------------------------------------------------------------------------------------------

#---------------------------------adding multi-labels value for each movie-----------------------------------------------------

copyGenres = []
for i in range(len(genres)):
    copyGenres.append(genres)

for i in range(len(copyGenres)):
    copyGenres[i] = []


for i in range(df.shape[0]):
    genresOfaMovie = (df.iat[i,2]).split("|")
    genreFoundAtIndexes = []
    for j in range(len(genresOfaMovie)):

        for k in range(len(genres)):
            
            if genresOfaMovie[j] == genres[k]:
                index = genres.index(genres[k])
                copyGenres[index].append(1)
                genreFoundAtIndexes.append(index)
                break
                
    for l in range(len(copyGenres)):
        if l not in genreFoundAtIndexes:
            copyGenres[l].append(0)


#------------------------------------------dropping old genres column----------------------------------------            
df = df.drop(columns = ['genres'])

#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------adding columns to data frame--------------------------------------------            
for i in range(len(genres)):
    df[str(genres[i])] = copyGenres[i]
#---------------------------------------------------------------------------------------------------------------               
                

#---------------------------------------------------------------------------------------------------------------

#----------------------------------Necessary DataFrame---------------------------------------------------------
titleFilmDF = df #------listing the movies and their corresponding genre affiliation-----#
ratingData = pd.read_csv('ratings.csv')
userDF = pd.DataFrame(ratingData) #--------the ratings of a single user for the movies he watched---------
#-------------------------------------------------------------------------------------------------------------
clusterScore = []
def clusterFilms(titleFilmDF):
    
    random.seed(123)
    i = 1
    titleFilmDF = titleFilmDF.drop(labels=['movieId','title'],axis=1)
    while True:
        random.seed(123)
        i += 1
        movieCluster = KMeans(n_clusters=i, random_state=0).fit(titleFilmDF)
        movieCluster2 = KMeans(n_clusters=i+1, random_state=0).fit(titleFilmDF)
        
        movieClusterlabels = movieCluster.labels_
#        movieCluster2labels = movieCluster2.labels_
#        
#        movieClusterDistance = KMeans(n_clusters=i, random_state=0).fit_transform(titleFilmDF)
#        movieCluster2Distance = KMeans(n_clusters=i+1, random_state=0).fit_transform(titleFilmDF)
        
#        movieClusterSilhouetteScore = silhouette_score(titleFilmDF,movieClusterlabels,metric='euclidean', sample_size=None, random_state=None)
#        movieCluster2SilhouetteScore = silhouette_score(titleFilmDF,movieCluster2labels,metric='euclidean', sample_size=None, random_state=None)
#        print(movieCluster2SilhouetteScore)
        
#        movieClusterTotalWithness = 0
#        for i in range(len(movieClusterlabels)):
#            movieClusterTotalWithness += ( (movieClusterDistance[i][movieClusterlabels[i]]) )**2
#            
#        movieCluster2TotalWithness = 0
#        for i in range(len(movieCluster2labels)):
#            movieCluster2TotalWithness += ( (movieCluster2Distance[i][movieCluster2labels[i]]) )**2
#            
#        decision = (movieClusterTotalWithness-movieCluster2TotalWithness)/ (movieClusterTotalWithness)
#        
#        if decision < 0.2:
#            
#            break
#        print("For k=",i)
#        print("Score: ",movieClusterSilhouetteScore)
#        clusterScore.append(movieClusterSilhouetteScore)
#        if movieClusterSilhouetteScore > 1:
#            break
        
        decision = (movieCluster.inertia_ - movieCluster2.inertia_)/movieCluster.inertia_
#        print(decision)
        
        if decision < 0.03:
            print("Final K Value is ",i)
            print()
            movieClusterSilhouetteScore = silhouette_score(titleFilmDF,movieClusterlabels,metric='euclidean', sample_size=None, random_state=None)
            print("Silhouette Score is ",movieClusterSilhouetteScore)
            print()
            break
        
    return movieCluster

def getUserInfo(dat , iD):
    a = dat.loc[dat['userId'] == iD ].drop(labels=['userId'],axis=1).reset_index(drop=True)
    clusters = np.zeros(len(a))
    activeUser = a.assign(cluster = clusters ).astype(int)
    activeUser = activeUser.rename(columns={"movieId" : "itemid"})
    activeUser = activeUser.sort_values("itemid")
    return activeUser

def setUserFilmCluster(movieCluster, activeUser):
    movieId = titleFilmDF['movieId']
    movieCluster = movieCluster.labels_
    data = {"movie_id":movieId,"cluster":movieCluster}
    df1 = pd.DataFrame(data)
    clusterNo = activeUser.merge(df1, how='inner', left_on='itemid', right_on='movie_id')
    activeUser["cluster"] = clusterNo["cluster_y"]
    return activeUser
    
def getMeanClusterRating(movieCluster, activeUser):
#    like = activeUser.groupby(["rating"]).mean()
    like = activeUser.groupby(["cluster"]).mean().drop(labels=["itemid"],axis=1).reset_index(drop=True)
    clusterNo=list()
    for i in range(len(like)):
        clusterNo.append(i)
    
    like = like.assign(cluster=clusterNo)
    maximumRating = max(like["rating"])
    if maximumRating < 3:
        like = 0
    else:
        maximumRating = max(like["rating"])
        vector = like.loc[like['rating'] == maximumRating]
        like = vector.iloc[0][1].astype(int) 
    
    return like
    
def getGoodFilms(like, movieCluster, titleFilmDF):
    movieId = titleFilmDF['movieId']
    movieCluster = movieCluster.labels_
    data = {"movie_id":movieId,"cluster":movieCluster}
    df1 = pd.DataFrame(data)
#    dirName = 'clustersAssignment'+ ' '+str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
     
#    try:
#        # Create target Directory
#        os.mkdir(dirName)
#        
#    except FileExistsError:
#        print("Directory " , dirName ,  " already exists")
#    #    print(df)
#    p = Path(dirName)
#    fileName = 'clusters' + '.csv' 
#    df1.to_csv(Path(p, fileName))
    
    if like == 0:
        recommend = titleFilmDF.random(n = 100)
    else:
        recommend = df1.loc[df1['cluster'] == like]
    
    
    return recommend

def getRecommendedFilms(titleFilmDF, userDF, userid):
    movieCluster = clusterFilms(titleFilmDF)
    activeUser = getUserInfo(userDF, userid)
    activeUser = setUserFilmCluster(movieCluster, activeUser)
    like = getMeanClusterRating(movieCluster, activeUser)
    recommend = getGoodFilms(like, movieCluster, titleFilmDF)
    watchedMovie = activeUser["itemid"]
    recommend = recommend[~recommend["movie_id"].isin(watchedMovie)].reset_index(drop=True) 
    movieTitleFrame = recommend.merge(titleFilmDF, how='inner', left_on='movie_id', right_on='movieId')
    movieTitle = movieTitleFrame["title"]
    recommend["movietitle"] = movieTitle 
    
    recommendFrame = recommend.merge(userDF, how='inner', left_on='movie_id', right_on='movieId')
    like = recommendFrame.groupby(["movie_id"]).mean().sort_values(by = ["rating"],ascending = False).reset_index(drop=True)
    likeWithMovieTitleFrame = recommend.merge(like, how='inner', left_on='movie_id', right_on='movieId')
    likeWithMovieTitle = likeWithMovieTitleFrame['movietitle']
    like["movietitle"] = likeWithMovieTitle
    return like
    
    


def suggestFilms(titleFilmDF, userDF, userid, no_films):
    userPresentEmpty = userDF[userDF['userId'].isin([userid])].empty

    if userPresentEmpty == False:
        suggestions = getRecommendedFilms(titleFilmDF, userDF, userid)
        #    print(len(suggestions))
        if len(suggestions) < no_films:
        #        print(len(suggestions))
            print("Try with some less number of Movies...")
        else:
            suggestions = suggestions[0:no_films]
        #        suggestions = suggestions.iloc[np.random.permutation(len(suggestions))].reset_index(drop=True)
            print("You may also like these movies.....")
            print()
            for i in range(len(suggestions)):
                print(suggestions["movietitle"][i])
    else:
        suggestions = titleFilmDF['title']
        suggestions = suggestions.iloc[np.random.permutation(len(suggestions))].reset_index(drop=True)  
        print("As You have not rated any movies,You may like these movies.....")
        print()
        suggestions = suggestions[0:no_films]
        for i in range(len(suggestions)):     
            print(suggestions.iloc[i])
        

    


suggestFilms(titleFilmDF,userDF, 5, 10) #---parameter1->listofMoviesWithGenres,parameter2->usersRatings,parameter3->userId,parameter4->nofFilmsToSuggest



    
