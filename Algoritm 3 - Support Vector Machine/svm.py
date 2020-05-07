import sys
import json
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pdb

def norm(w):
    return np.linalg.norm(w)
arange = np.arange
dot = np.dot

style.use('ggplot')


def find_option_with_smallest_norm(options_dict):
    print('Given dict:')
    print(options_dict)
    norms = sorted([n for n in options_dict])
    print(norms)
    return options_dict[norms[0]]

class SupportVectorMachine:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)

    def train(self, data):
        self.data = data
        print('Got training data as')
        print(data)
        '''
        opt_dict = dict with keys ||w|| and values {'w' : w, 'b' : b}
        Pentru fiecare key ||w||, retine w si b gasit.
        Atentie! O sa fie gasite mai multe variante pentru w si b ale aceluiasi ||w||.
        O sa se suprascrie ||w|| de mai multe ori. De aceea le retinem intr-un dictionar.
        '''
        opt_dict = {}

        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]

        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        print('feature vals:')
        print(self.max_feature_value)
        print(self.min_feature_value)
        
        '''
        Valori euristice pentru marimile pasilor la fiecare dintre cele 3 trepte
        '''
        step_sizes = [0.5, 0.25, 0.01]

        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value * 5
        print(f'Started with latest_optimum={latest_optimum}')
        

        '''
        Facem 3 iteratii (in 3 trepte).
        La inceput, treapta (step) este mai mare, ca sa putem face pasi mai mari.
        Pentru fiecare treapta, incercam sa gasim un 'w' a carui norma este cat mai mica.
        
        w va fi mereu de forma [a, a]
          (pe doua coordonate, pentru ca acest SVM este pe doua coordonate)

        (1) Pornim cu w de la o valoare euristica [a, a]
        In acest algoritm, luam latest_optimum pe post de a,
          astfel incat sa incepem de la un punct departat,
          mai mare decat oricare alt punct din setul de date.
          (acest latest_optimum)
        
        Pentru fiecare w prin care iteram:
          Incercam toate b-urile intre doua valori euristice
            (cu valori euristice pentru marimea pasilor iteratiei lui b)
            Cautam toate combinatiile indicilor lui w, astfel incat
              sa satisfaca conditia: (y * (dot(w, x) + b) >= 1
              pentru TOATE punctele din data set
            Combinatiile lui w cu forma [a,a] pot fi:
              [a,a] [-a,a] [-a,-a] [a,-a]
            Cand gasim o combinatie [w, b] care satisface conditia de mai sus,
              o retinem si ii retinem norma.
            Note: daca se suprascrie
        
        Dintre toate normele gasite, o obtinem pe cea mai mica si trecem la treapta urmatoare.
        Ne intoarcem la pasul (1) si il initializam pe w cu o valoare putin mai mare decat
          cea gasita.
        Ne oprim cand nu mai avem trepte de parcurs.

        '''

        for step in step_sizes: # 3 trepte in total

            w = np.array([latest_optimum, latest_optimum])  # initializam w cu forma [a,a]
            tried_all_options = False

            while not tried_all_options:
                print(f'Trying w = {w}')
                b_from = self.max_feature_value * b_range_multiple * (-1) # Exact cum am zis de domeniu pentru w mai sus, asa facem si pentru b
                b_upto = self.max_feature_value * b_range_multiple      # Doar ca domeniul lui b e mai mare si luam si pasi mai mari
                b_step_size = step * b_multiple

                for b in arange(b_from, b_upto, b_step_size):   # De la b_from, pana la b_upto, din b_step_size in b_step_size

                    for transformation in transforms:   # Luam fiecare dintre cele 4 variante posibile ale lui w (cu + si - la termeni)
                        w_t = w * transformation
                        found_option = True

                        for y in self.data: # y este -1 sau 1; data[y] contine o lista de x
                            for x in self.data[y]: # x este o lista de coordonate
                                if not y * (dot(w_t, x) + b) >= 1:
                                    found_option = False

                        if found_option:  # Daca am gasit o astfel de optiune, o retinem
                            opt_dict[norm(w_t)] = {
                                'w': w_t,
                                'b': b
                            }
                            break

                if w[0] < 0:
                    tried_all_options = True
                    print('Optimized step ' + str(w))
                    break
                else:
                    w = w - step

            opt_choice = find_option_with_smallest_norm(opt_dict)
            self.w = opt_choice['w']
            self.b = opt_choice['b']
            latest_optimum = opt_choice['w'][0] + step * 2

    def predict(self, features):
        colors = {1:'r', -1: 'b'}
        classification = sign(dot(np.array(features), self.w) + self.b) # Calculam clasificarea dupa formula
        if classification != 0:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=colors[classification])
        return classification

    def visualize(self):
        colors = {1:'r', -1: 'b'}
        [[self.ax.scatter(x[0], x[1], s=100, color=colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x,w,b,v):
            return (-w[0]*x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        w = self.w
        b = self.b

        print(f'Plotting w={w}')

        # (w dot x + b) = 1
        # Plotting positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, w, b, 1)
        psv2 = hyperplane(hyp_x_max, w, b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # (w dot x + b) = -1
        # Plotting negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, w, b, -1)
        nsv2 = hyperplane(hyp_x_max, w, b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # (w dot x + b) = 0
        # Plotting middle support vector hyperplane
        hp1 = hyperplane(hyp_x_min, w, b, 0)
        hp2 = hyperplane(hyp_x_max, w, b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [hp1, hp2])
        
        plt.show()


data_dict = None
with open('Datasets/svm_points.json', 'r') as file:
    points = json.loads(file.read())
    data_dict = {
        -1: [],
         1: []
    }
    for point in points:
        data_dict[point['label']].append(point['x'])
    data_dict = {
        -1: np.array(data_dict[-1]),
         1: np.array(data_dict[1])
    }

print('Read json as:')
print(data_dict)

'''
data_dict = {
    -1: np.array([[1,7], [2,8], [3,8]]),
     1: np.array([[5,1], [6,-1], [7,3]])
}
'''

svm = SupportVectorMachine()

svm.train(data_dict)

svm.visualize()