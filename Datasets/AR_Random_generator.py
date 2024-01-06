#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Shubham S. Raheja (shubhamsraheja@gmail.com)
Code Description: A python code to generate AR (Auto-regressive) and Random Series for causality identification.
"""
import os
import numpy as np
import random as rd
import pickle
from scipy import stats

class CreateDatasets:
    def __init__(self, coeff, order, rows, length):
        self.coeff = coeff
        self.order = order
        self.rows = rows
        self.length = length

    # Simulation of AR processes - Creating AR datatset
    def create_ar(self):
        """
        Dataset creation for AR process.

        Parameters
        ----------
        coeff  : scalar, float
            Coefficient at lag i time step(s). 
            AR coefficient for each simulation is randomly chosen such that a k ∈ U (0.8, 0.9)
        order  : scalar, int
            Order of the AR process. 
            Randomly chosen between 1 and 20.
        length : scalar, int
            Number of data points required.    
        Returns
        -------
        array: array, 1D, float
            A single AR vector of length as specified by user.
    
        """
        array = np.ones((1,self.length))
        for i in range(self.order):
            array[0,i] = np.random.default_rng().normal(loc=0, scale=0.1) # Initial values are selected from normal(0,0.01)
        for i in range(self.length-self.order):
            err=np.random.default_rng().normal(loc=0, scale=0.1) # Error is set from normal(0,0.01)
            array[0,i+self.order] = (self.coeff*array[0,i]) + err
        return (array)

    def create_random(self):
        """
        Dataset creation for Random process.

        Parameters
        ----------
        rows  : scalar, float
            Number of rows required in the random dataset matrix
        length : scalar, int
            Number of data points required.    
        Returns
        -------
        array: matrix, 2D, float
            A matrix of Random sequences of length as specified by user.
    
        """  
        rand_array=np.ones((1,self.length))
        low = 0
        high = 1
        mean = 0
        stddev = 0.1
        rand_array = stats.truncnorm.rvs(low, high,loc = mean, scale = stddev,size = self.length)
        return rand_array

    def save_object(self, obj, filename):
        with open(filename, 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def save_datasets(self, save_name_ar, save_name_rand):

        ar_dataset= np.ones((self.rows,self.length)) # ar_dataset is the matrix that would contain all the ar vectors as rows
        for i in range(self.rows):
            ar_dataset[i,:] = self.create_ar()
        self.save_object(ar_dataset, save_name_ar)

        rand_dataset = np.ones((self.rows,self.length))
        for i in range(self.rows):
            rand_dataset[i,:] = self.create_random()
        self.save_object(rand_dataset, save_name_rand)       

if __name__== "__main__":
    coeff = rd.uniform(0.8,0.9) # Coefficient for ith vector
    order = rd.randint(1,20) # Order of the AR process is randomly selected between 1 and 20
    rows = 1250
    length = 50 # Pick from 50, 75 and 100
    try:
        save_dir = f"path/Length_{length}"
    except:
        save_dir = f"path/Length _{length}"
        os.makedirs(save_dir)
    save_name_ar = os.path.join(save_dir, "ar_dataset.pkl")
    save_name_rand = os.path.join(save_dir, "rand_dataset.pkl")


    create_datasets = CreateDatasets(coeff, order, rows, length)
    create_datasets.save_datasets(save_name_ar, save_name_rand)


        




