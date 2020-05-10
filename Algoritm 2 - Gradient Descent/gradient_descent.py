import numpy
import matplotlib.pyplot as pyplot
import pandas
import json
import os.path as path
import statistics
import time

def get_points():   # Get the points from Datasets/points.json
    dir_path = path.dirname(path.realpath(__file__))
    datasets_folder = path.join(dir_path, 'Datasets')
    points_path = path.join(datasets_folder, 'points.json')
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
############################ Gradient Descent ###########################
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




def gradient_descent_for_intercept(points, m, show=False):

    '''
    Bazat pe ecuatia dreptei.
    Obtine y stiind ecuatia dreptei si un x dat.
    Panta (m) este data dupa ce e calculata cu Least Squares.
    Interceptul (c) este calculat prin interatii, mai jos.
    '''
    def predict_y(x, c):
        return m * x + c

    '''
    Eroarea prezicerii functiei de mai sus.
    Cu cat mai aproape de 0, cu atat mai bine.
    '''
    def residual_error(point, c):
        return point['y'] - predict_y(point['x'], c)

    '''
    Functia loss este o parabola, un polinom de gradul 2 in functie de c.
    Minimul acesei functii inseamna cel mai bun c, astfel incat erorile totale
      sa fie cat mai mici.
    '''
    def loss(c):
        return sum([ residual_error(point, c)**2 for point in points ])

    '''
    Din cauza ca loss function este de gradul 2, derivata este de gradul 1,
      adica o dreapta.
    Derivata intr-un punct de pe loss function are o panta.
    Cu cat panta se apropie de 0, cu atat inseamna ca ne apropiem de minimul functiei loss.
    '''
    def loss_deriv(c):
        return sum([ (-2)*residual_error(point, c) for point in points ])

    ######################### Algoritm #####################

    '''
    Optional, putem pune pe grafic si graficul functiei loss
    '''
    def plot_helpers():
        cs = numpy.linspace(-5, 5, 100)
        losses = [loss(c) for c in cs]
        plot_points([{
            'x' : cs[i],
            'y' : losses[i]
        } for i in range(len(cs))], False)

        losses = [loss_deriv_slope(c) for c in cs]
        plot_points([{
            'x' : cs[i],
            'y' : losses[i]
        } for i in range(len(cs))], False, 'ro')
    

    '''
    Pornim cu c de la o valoare arbitrara in 'stanga' punctului de minim al functiei loss.
    La fiecare iteratie, ne vom apropia cu panta derivatei de 0.
    Folosim aceasta panta ca sa ajustam interceptul (c).
    Ne oprim cand am ajuns foarte aproape de 0.
    '''
    c = 0
    while True:
        print(c, loss_deriv(c))
        if abs(loss_deriv(c)) < 0.1:    # Verificăm dacă ne-am apropiat suficient de tare
            break
        c = c - loss_deriv(c) * 0.01        
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
b = gradient_descent_for_intercept(points, m)

plot_line(m, b)
pyplot.show()






