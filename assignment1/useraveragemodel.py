# -*- coding: utf-8 -*-
from model import Model


class UserAverageModel(Model):

    def create_model(self):
        self.calculate_average_user_ratings()

    def rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)

        if user_id in self.user_averages:
            return self.user_averages[user_id]
        else:
            print("No data, returning global average")
            return self.global_average

    def calculate_average_user_ratings(self):
        self.user_averages = dict()

        user_rating_dict = dict()

        global_sum = 0.0
        total_ratings = 0

        for sset in self.training_set:
            for rating in sset:
                global_sum += float(rating[2])
                total_ratings += 1

                user_id = int(rating[0])
                u_rating = int(rating[2])

                if user_id in user_rating_dict:
                    user_rating_dict[user_id].append(u_rating)
                else:
                    user_rating_dict[user_id] = [u_rating]

        self.global_average = float(global_sum) / total_ratings

        for user_id in user_rating_dict.keys():
            self.user_averages[user_id] = float(sum(user_rating_dict[user_id])) / len(user_rating_dict[user_id])
