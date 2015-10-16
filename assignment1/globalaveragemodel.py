# -*- coding: utf-8 -*-
from model import Model


class GlobalAverageModel(Model):

    def create_model(self):
        self.calculate_average_rating()

    def rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)

        return self.average

    def calculate_average_rating(self):
        sum_rating = 0
        total_ratings = 0
        for split in self.training_set:
            total_ratings += len(split)
            for rating in split:
                sum_rating += int(rating[2])

        self.average = float(sum_rating) / total_ratings
