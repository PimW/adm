# -*- coding: utf-8 -*-
from model import Model
import numpy as np


class CFUU(Model):
    
    def __init__(self, users, similarity_measure, top_k, training_matrix):
        self.users = users
        self.similarity_measure = similarity_measure
        self.top_k = top_k
        self.training_matrix = training_matrix
        self.create_model()
        
    def create_model(self):
        pass
    
    def rating(self, user_id, movie_id):
        rating = 0
        
        user_similarities = {}

        user1_vector = self.training_matrix[user_id, :]
        
        users = []

        index = 0
        for rating in self.training_matrix[:, movie_id]:
            if rating > 0:
                users.append(index)
            index += 1
        
        for user in users:
            user2_vector = self.training_matrix[user, :]

            similarity = self.similarity_measure(user1_vector, user2_vector)
            user_similarities[user] = similarity

        sorted_users = sorted(user_similarities, key=lambda x: user_similarities[x], reverse=True)
        top_users  = sorted_users[:self.top_k]
        
        total_rating = 0
        for user in top_users:
            rating = self.training_matrix[user, movie_id]
            total_rating += rating
        predicted_rating = total_rating / self.top_k
        print(predicted_rating)
        return predicted_rating