# -*- coding: utf-8 -*-
from model import Model
import numpy as np
import random


class GravityModel(Model):
    def __init__(self, data, movies, users, training_set, num_iterations=100):
        super().__init__(movies, users, training_set)

        self.num_iterations = num_iterations
        self.X = data.create_matrix()  # actual matrix
        self.U = self.create_user_matrix(users)  # matrix U of size (i,K)
        self.M = self.create_movie_matrix(movies)  # matrix M of size (K,j)

        #TODO: change values
        self.K = 0  # amount of features to consider
        self.rate = 0  # learning rate
        self.reg = 0  # regularisation factor

    def create_user_matrix(self, users):
        """
        Create a matrix of size i*K where each row contains the features for each user
        Weights are initialised randomly
        :return: Matrix (i,K) containing initialised features
        """
        U = np.empty(len(users), self.K)
        for i in range(0, len(users)):
            row = []
            for k in range(0, self.K):
                row.append(random.random())
            np.append(U, row, axis=0)
        return U

    def create_movie_matrix(self, movies):
        """
        Create a matrix of size K*j where each column contains the features for each movie
        Weights are initialised randomly
        :return: Matrix (K,j) containing initialised features
        """
        M = np.empty(self.K, len(movies))
        for i in range(0, len(movies)):
            row = []
            for k in range(0, self.K):
                row.append(random.random())
            np.append(M, row, axis=0)
        return M

    def train_model(self):
        """
        Iterate with a number of iterations or until the value of the RMSE doesn't change for 2 iterations
        During each iteration for each item in the matrix (not in the probe subset) calculate the error(i,j)
            then compute the gradient of the error^2 using formula 5 pg. 24
            after that update the values of U[i,k] and M[k,j] using formulas 6 and 7 pg. 24
        after each iteration compute the RMSE on the probe subset
        """
        values = []
        for it in range(0, self.num_iterations):  #TODO: change to until no change in RMSE
            print("Finished iteration %d/%d" % (it, self.num_iterations))
            for val in np.nonzero(self.X):
                for k in range(0, self.K):
                    self.update(val[0], val[1], k)

    def update(self, i, j, k):
        error_ij = self.training_error(i, j)
        ik = self.U.item(i, k)
        kj = self.M.item(k, j)

        new_ik = ik + self.rate * (2 * error_ij * kj - self.reg * ik)
        new_kj = kj + self.rate * (2 * error_ij * ik - self.reg * kj)
        self.U[i, k] = new_ik
        self.M[j, k] = new_kj

    def predict_rating(self, i, j):
        """
        Calculate the predicted rating using the U and M matrices using formula 3 pg. 24
        :return: Estimated rating
        """
        estimate = 0
        for k in range(0, self.K):
            estimate += (self.U.item(i, k) * self.M.item(k, j))  # TODO: change this to use proper numpy matrix calls
        return estimate

    def training_error(self, i, j):
        """
        Calculate the error of the predicted rating using the actual rating matrix and the prediction
        :return: Error of the predicted rating
        """
        error = self.X.item(i, j) - self.predict_rating(i, j)  # TODO: change this to use proper numpy matrix calls
        return error
