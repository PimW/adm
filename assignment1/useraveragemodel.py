# -*- coding: utf-8 -*-
from model import Model
import numpy as np

class UserAverageModel(Model):
    
    def __init__(self, movies, users, training_matrix):
        self.movies = movies
        self.users = users
        self.training_matrix = training_matrix
        self.create_model()
        
    def create_model(self):
        # Sum over all ratings for each user
        rating_sums = np.sum(self.training_matrix, axis=1)
        
        # Count of all non-zero ratings for each movie
        rating_counts = np.sum(self.training_matrix > 0, axis=1)
        
        # Get global_average before altering data
        self.global_average = np.sum(rating_sums) / np.sum(rating_counts)
        
        # Prevent division by zero / nan
        rating_counts[rating_counts == 0] = 1
        rating_counts[np.isnan(rating_counts)] = 1
        
        self.ratings = rating_sums / rating_counts
    
    def rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)
        
        result = self.ratings[user_id]
        if result > 0:
            return result
        else:
            return self.global_average