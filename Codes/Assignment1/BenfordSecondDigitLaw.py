#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Authors: Bharat Chandra Mukkavalli (ES19BTECH11016), Krishn Vishwas Kher (ES19BTECH11015), Sayantan Biswas (AI19BTECH11015).
'''

import pandas
import numpy
from scipy.stats import chisquare

def probabilities(data):
    
    probs = [0 for _ in range(10, 100)]
    for number in data:
        if len(str(number)) >= 2:
            first_digit = str(number)[0]
            second_digit = str(number)[1]
            probs[10*int(first_digit)+int(second_digit)-10] += 1
    
    for index in range(len(probs)):
        probs[index] /= len(data)
        
    return probs

def MAD(observed, expected):
    
    if len(observed) != len(expected):
        raise Exception("Error! The number of observed and expected values do not match (observed list size: %d, expected list size: %d).", len(observed), len(expected))
    elif len(observed) != 90:
        raise Exception("Error: The list of observed probabilites must have exactly 9 observations (it currently has %d observations).", len(observed))
        
    sum = 0.0
    for index in range(len(observed)):
        sum += abs(observed[index] - expected[index])
    
    sum /= len(observed)
    
    return sum
        


dataset = pandas.read_csv('uniform.csv')
dataset.data = dataset.data.astype('int64')
dataset = list(dataset.data)

benford_probabilities = [numpy.log10(1.0 + 1.0/x) for x in range(10, 100)]
#print(benford_probabilities)
observed_probabilities = probabilities(dataset)
print(observed_probabilities)
print(benford_probabilities)
kai2, p_val = chisquare(observed_probabilities, benford_probabilities)
print("The MAD value between the expected and observed probabilites is:", MAD(observed_probabilities, benford_probabilities),"\b.")
print("The p value upon running the χ2-square test is:", p_val,"\b.")
