# -*- coding: utf-8 -*-

# Abstract model
class Model(object):

    def __init__(self, movies, users, training_set):
        self.movies = movies
        self.users = users
        self.training_set = training_set
        self.create_model()

    def create_model(self):
        return

    def rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)

        return -100