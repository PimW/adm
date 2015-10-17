# -*- coding: utf-8 -*-

import re
import random
import numpy as np


class Data(object):
    def __init__(self, data_path):
        self.data_path = data_path

        self.ratings_filename = 'ratings.dat'
        self.movies_filename = 'movies.dat'
        self.users_filename = 'users.dat'
        self.encoding = 'utf-8'

        self.movies = self.read_movies()
        self.ratings = self.read_ratings()
        self.users = self.read_users()

    def read_ratings(self):
        f = open(self.data_path + self.ratings_filename, 'r', encoding=self.encoding, errors='replace')
        text = f.read()
        ratings = re.findall('(\d+)::(\d+)::(\d+)::(\d+)', text)
        return ratings

    def read_movies(self):
        f = open(self.data_path + self.movies_filename, 'r', encoding=self.encoding, errors='replace')
        text = f.read()
        ratings = re.findall('(\d+)::(.+)::(.+)', text)
        return ratings

    def read_users(self):
        f = open(self.data_path + self.users_filename, 'r', encoding=self.encoding, errors='replace')
        text = f.read()
        users = re.findall('(\d+)::(.+)::(.+)::(.+)::(.+)', text)
        return users

    def split_ratings(self, ratings):
        splits = [[], [], [], [], []]
        for rating in ratings:
            r = random.randint(0,4)
            splits[r].append(rating[0:3])

        return splits

    def generate_sets(self, splits):
        sets = []
        for split in splits:
            sset = list(splits)
            sset.remove(split)
            sets.append(sset)

        return sets

    def five_fold(self, model):
        sets = self.generate_sets()
        models = []
        for sset in sets:
            models.append(model(self.movies, self.users, sset))


    def create_matrix(self, movies, users, sset):
        movie_count = int(self.movies[-1][0]) + 1
        user_count = len(self.users) + 1

        matrix = np.full((user_count, movie_count), 0)

        for split in sset:
            for rating in split:
                user_id = int(rating[0])
                movie_id = int(rating[1])
                u_rating = int(rating[2])
                matrix[user_id, movie_id] = u_rating

        return matrix
