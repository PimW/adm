# -*- coding: utf-8 -*-
from model import Model
import numpy as np


class MovieAverageModel(Model):

    def __init__(self, movies, users, training_matrix):
        self.movies = movies
        self.users = users
        self.training_matrix = training_matrix
        self.create_model()

    def create_model(self):
        return

    def rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)

        rating_sum = np.sum(self.training_matrix[movie_id])

        number_of_ratings = np.count_nonzero(self.training_matrix[movie_id])

        return float(rating_sum) / number_of_ratings