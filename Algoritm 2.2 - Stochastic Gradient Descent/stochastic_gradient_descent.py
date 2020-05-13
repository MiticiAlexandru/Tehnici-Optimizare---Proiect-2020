import numpy
import random
import matplotlib.pyplot as pyplot
import pandas
import json
import os.path as path
import statistics
import time

def get_points():   # Get the points from Datasets/points.json
    dir_path = path.dirname(path.realpath(__file__))
    datasets_folder = path.join(dir_path, 'Datasets')
    points_path = path.join(datasets_folder, 'stochastic_points.json')
    points = None
    with open(points_path, 'r') as file:
        points = json.loads(file.read())
    return points

def plot_points(points, show=True, color='go'): # Plots the points
    xs = [point['x'] for point in points]
    ys = [point['y'] for point in points]

    pyplot.plot(xs, ys, color)
    if show:
        pyplot.show()

def plot_line(slope, intercept):
    xs = numpy.linspace(0, 25, 100)
    ys = slope * xs + intercept
    pyplot.plot(xs, ys, '-b')



#########################################################################
############################ Stochastic Gradient Descent ###########################
#########################################################################

# Uses the Least Squares algorithm/formula to find the optimal slope for Linear Regression.
def get_slope_by_least_squares(points):
    xs = [point['x'] for point in points]
    ys = [point['y'] for point in points]

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    for point in points:
        point["x_prime"] = point['x'] - mean_x
        point["y_prime"] = point['y'] - mean_y
    
    numarator = sum([point["x_prime"] * point['y_prime'] for point in points])
    numitor = sum([point["x_prime"] ** 2 for point in points])
    m = numarator / numitor
    return m




def stochastic_gradient_descent_for_intercept(points, m, show=False):

    '''
    Bazat pe ecuatia dreptei.
    Obtine y stiind ecuatia dreptei si un x dat.
    Panta (m) este data dupa ce e calculata cu Least Squares.
    Interceptul (c) este calculat prin interatii, mai jos.
    '''
    def predict_y(x, c):
        return m * x + c

   
    def stochastic_loss(c):
        r = random.choice(points)
        return 


    '''
    Obtinem panta printr-un calcul simplu.
    '''
    def stochastic_loss(c):
        r = random.choice(points)
        return (-2) * (r['y'] - c - m * r['x'])

    ######################### Algoritm #####################

    '''
    Pornim cu c de la o valoare arbitrara in 'stanga' punctului de minim al functiei loss.
    La fiecare iteratie, loss functionul devine din ce in ce mai mic.
    Ca urmare, c devine din ce in ce mai aproape de solutie.
    Ne oprim cand am ajuns foarte aproape de 0.
    '''
    c = 0
    while True:
        step_size = stochastic_loss(c) * 0.005
        c = c - step_size
        print(c, step_size)
        if abs(step_size) < 0.0001:
            break
        time.sleep(0.2)
        
    
    if show:
        plot_helpers()
        pyplot.plot([c], [loss(c)], 'bo')
    return c


points = get_points()

pyplot.ylim(0, 10)
pyplot.xlim(0, 10)

plot_points(points, show=False)
m = get_slope_by_least_squares(points)
b = stochastic_gradient_descent_for_intercept(points, m)

plot_line(m, b)
pyplot.show()






