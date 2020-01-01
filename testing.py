# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:33:15 2019

@author: Rahul
"""
#import numpy as np
##a = ['b','c','d']
##
##copya = []
##
##for i in range( len(a) ):
##    copya.append( a[i] )
##
##for i in range(len(copya)):
##    copya[i] = []
##    
##no = 'c'
##
##index = a.index(no)
##copya[index].append(0)
##copya[index].append(0)
##copya[index].append(0)
##copya[0].append(1)
##copya[0].append(1)
##copya[0].append(1)
##copya[2].append(1)
##copya[2].append(1)
##print(copya)
#
##a = np.NaN
##print(a)
#
#a = [[2,3,4],[1,4,6],[9,7,6]]
#b =  [2,3,4]
#c= [1,4,6]
#d = [9,7,6]
#list_tuples = list(zip(b,c,d))
#print(list_tuples)
#list_of_tuples = [*zip(*a)]
#print(list_of_tuples)


from imdb import IMDb
ia = IMDb()
moviesId = [114709,113497,113228,114885,113041,113277,114319]

for j in range(len(moviesId)):
    movie = ia.get_movie(moviesId[j])
    print("Name of the movie: ", movie)
    for i in movie['director']:
        print("Director: ", i)
        director = ia.search_person(i["name"])[0]
        ia.update(director)
    
