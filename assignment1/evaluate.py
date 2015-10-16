# -*- coding: utf-8 -*-
import math

def check_error_RMSE(model, test_split):
    error = 0.0

    for rating in test_split:
        user_id = int(rating[0])
        movie_id = int(rating[1])

        predicted_rating = model.rating(user_id, movie_id)
        actual_rating = float(rating[2])
        error += math.pow(predicted_rating - actual_rating, 2)

    error = math.sqrt(error / len(test_split))

    return error

def check_error_MAE(model, test_split):
    error = 0.0

    for rating in test_split:
        user_id = int(rating[0])
        movie_id = int(rating[1])

        predicted_rating = model.rating(user_id, movie_id)
        actual_rating = float(rating[2])
        error += math.fabs(predicted_rating - actual_rating)

    return error / len(test_split)

