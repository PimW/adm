# -*- coding: utf-8 -*-

import random
import numpy as np

from globalaveragemodel import GlobalAverageModel
from movieaveragemodel import MovieAverageModel
from useraveragemodel import UserAverageModel
from gravitymodel import GravityModel

from evaluate import check_error_MAE, check_error_RMSE
from data import Data

data_path = '/home/pimw/PycharmProjects/adm/assignment1/dmdata/'

# Create a data object holding all information
data = Data(data_path)


def split_ratings(ratings):
    splits = [[], [], [], [], []]
    for rating in ratings:
        r = random.randint(0,4)
        splits[r].append(rating[0:3])
        
    return splits


def generate_sets(splits):
    sets = []
    for split in splits:
        sset = list(splits)
        sset.remove(split)
        sets.append(sset)
        
    return sets


def create_matrix(movies, users, sset):
    movie_count = int(movies[-1][0]) + 1
    user_count = len(users) + 1
    
    matrix = np.full((user_count, movie_count), 0)
   
    for split in sset:
        for rating in split:
            user_id = int(rating[0])
            movie_id = int(rating[1])
            u_rating = int(rating[2])
            matrix[user_id, movie_id] = u_rating
            
    return matrix


# Generate 5 splits of approx. same size
splits = split_ratings(data.ratings)

# Generate 5 unique sets containing 4 splits each
sets = generate_sets(splits)


def check_models(models):
    for x in range(0, len(models)):
        test_split = splits[x]
        model = models[x]
        rmse = check_error_RMSE(model, test_split)
        mae = check_error_MAE(model, test_split)
        print("Model " + str(x+1) + " RMSE: " + str(rmse) + " MAE: " + str(mae))

# In[784]:

# Global average model
models = []
for sset in sets:
    models.append(GlobalAverageModel(data.movies, data.users, sset))

#check_models(models)


models = []
# User average model
#for sset in sets:
#    models.append(UserAverageModel(data.movies, data.users, sset))

#check_models(models)

models = []
# Movie average model
for sset in sets:
    matrix = create_matrix(data.movies, data.users, sset)
#    models.append(MovieAverageModel(data.movies, data.users, matrix))
    
#check_models(models)


models = []
# Movie average model
for sset in sets:
    matrix = create_matrix(data.movies, data.users, sset)
    models.append(GravityModel(None, data.movies, data.users, matrix))

print("Checking gravitymodel")    
check_models(models)

matrix = create_matrix(data.movies, data.users, sets[0])
model = MovieAverageModel(data.movies, data.users, matrix)

rating_sum = 0.0
total_ratings = 0
for rating in model.training_matrix[1]:
    if rating > 0:
        total_ratings += 1
        rating_sum += rating


# In[ ]:

print(sum([164.0,160.0,201.0,187.0,176.0]) / 5)
x = 0
total = 0
for rating in data.ratings:
    user_id = int(rating[0])
    movie_id = int(rating[1])
    rating = int(rating[2])
    if(movie_id == 1):
        if rating > 0:
            x += rating
            total += 1
            
print(float(x) / total)



