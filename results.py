import ipdb
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import config as c

class Result:
    def __init__(self,s,i,b,x_error,x_total,rest,Uxf):
        self.spread=float(s)
        self.inertia=float(i)
        self.balancing=float(b)
        self.x_flow_error=float(x_error)
        self.x_total=float(x_total)
        self.rest=float(rest)
        self.Uxf=float(Uxf)

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

def plot_results(xs,ys,zs, x_label='X label', y_label='Y label',
                     z_label='Z label', plot_label='Plot label', c='b'):
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, 'o', c=c, marker='o', label=plot_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    # ax.legend()

if __name__ == "__main__":
    results_file = "results_4253648295_300towers.txt"

    spread_results = {}
    with open('%s/%s' % (c.DATA_DIR,results_file)) as f:
        for row in csv.reader(f, delimiter=','):
            s, i, b, x_error, x_total, rest, Uxf = row
            if s == 'spread':
                continue
            if s in spread_results:
                spread_results[s].append(Result(s, i, b, x_error, x_total, rest, Uxf))
            else:
                spread_results[s] = [Result(s, i, b, x_error, x_total, rest, Uxf)]

    mpl.rcParams['legend.fontsize'] = 10

    colors = ['b','g','r','c','m','y','k']

    # Plot a layer for each spread level
    fig = plt.figure()
    for ((k,v),c) in zip(spread_results.iteritems(),colors):
        xs,ys,zs = zip(*[(x.inertia,x.balancing,x.x_flow_error) for x in v])
        plot_results(xs,ys,zs, x_label='Inertia', y_label='Balancing',
                     z_label='Error', plot_label='Spread = %s' % k, c=c)
        plt.hold(True)

    plt.legend()
    plt.show()
    ipdb.set_trace()

