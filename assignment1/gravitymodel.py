# -*- coding: utf-8 -*-
from model import Model
import numpy as np


class GravityModel(Model):
    def __init__(self, data, movies, users, training_set, num_iterations=100):
        super().__init__(movies, users, training_set)

        self.num_iterations = num_iterations
        self.X = data.create_matrix()  # actual matrix
        self.U = self.create_user_matrix()  # matrix U of size (i,K)
        self.M = self.create_movie_matrix()  # matrix M of size (K,j)
        self.K = 0  # amount of features to consider
        self.rate = 0  # learning rate
        self.reg = 0  # regularisation factor

    def create_user_matrix(self):
        """
        Create a matrix of size i*K where each row contains the features for each user
        :return: Matrix (i,K) containing initialised features
        """
        return np.full()

    def create_movie_matrix(self):
        """
        Create a matrix of size K*j where each column contains the features for each movie
        :return: Matrix (K,j) containing initialised features
        """
        return np.full()

    def train_model(self):
        """
        Iterate with a number of iterations or until the value of the RMSE doesn't change for 2 iterations
        During each iteration for each item in the matrix (not in the probe subset) calculate the error(i,j)
            then compute the gradient of the error^2 using formula 5 pg. 24
            after that update the values of U[i,k] and M[k,j] using formulas 6 and 7 pg. 24
        after each iteration compute the RMSE on the probe subset
        """
        values = []
        for it in range(0, self.num_iterations):
            print("Finished iteration %d/%d" % (it, self.num_iterations))
            for val in values:
                self.update()

    def update(self):
        pass

    def predict_rating(self, i, j):
        """
        Calculate the predicted rating using the U and M matrices using formula 3 pg. 24
        :return: Estimated rating
        """
        estimate = 0
        for k in range(0, self.K):
            estimate += (self.U[i][k] * self.M[k][j])  # TODO: change this to use proper numpy matrix calls
        return estimate

    def training_error(self, i, j):
        """
        Calculate the error of the predicted rating using the actual rating matrix and the prediction
        :return: Error of the predicted rating
        """
        error = self.X[i][j] - self.predict_rating(i, j) # TODO: change this to use proper numpy matrix calls
        return error
