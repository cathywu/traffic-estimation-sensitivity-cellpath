import ipdb
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import config as c
from helpers import array

class Result:
    def __init__(self,s,i,b,x_error,x_total,rest,Uxf,x_error_indict=None,
                 x_true=None,x_est=None,f=None):
        self.spread=float(s)
        self.inertia=float(i)
        self.balancing=float(b)
        self.x_flow_error=float(x_error)
        self.x_total=float(x_total)
        self.rest=float(rest)
        self.Uxf=float(Uxf)
        self.x_error_indict=float(x_error_indict)
        self.x_true=array(np.fromstring(x_true[1:-1], sep=' '))
        self.x_est=array(np.fromstring(x_est[1:-1], sep=' '))
        self.f=array(np.fromstring(f[1:-1], sep=' '))

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

def plot_results_wireframe(ax,xs,ys,zs, x_label='X label', y_label='Y label',
                 z_label='Z label', plot_label='Plot label', c='b'):
    # ax = fig.add_subplot(111, projection='3d')
    XS,YS = sorted(list(set(xs))),sorted(list(set(ys)))
    X,Y = np.meshgrid(XS,YS)
    Z = np.empty_like(X)
    for (x,y,z) in zip(xs,ys,zs):
        Z[YS.index(y),XS.index(x)] = z
    ax.plot_wireframe(X, Y, Z, label=plot_label, color=c)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    # ax.legend()
    ipdb.set_trace()

if __name__ == "__main__":
    results_file = "results_4253648295_300towers.txt"
    # results_file = "results.txt"
    results_file = 'results_4253648295_375towers_take2.txt'

    spread_results = {}
    # with open('%s/%s' % (c.DATA_DIR,results_file)) as f:
    #     for row in csv.reader(f, delimiter=','):
    #         s, i, b, x_error, x_total, rest, Uxf = row
    #         if s == 'spread':
    #             continue
    #         r = Result(s, i, b, x_error, x_total, rest, Uxf)
    #         if s in spread_results:
    #             spread_results[s].append(r)
    #         else:
    #             spread_results[s] = [r]

    # FOR NEW EXPERIMENTS (new data format)
    with open('%s/%s' % (c.DATA_DIR,results_file)) as f:
        for row in csv.reader(f, delimiter='|'):
            s, i, b, x_error, x_total, rest, Uxf, x_error_indict, x_true, x_est, f = row
            if s == 'spread':
                continue
            r = Result(s, i, b, x_error, x_total, rest, Uxf,
                       x_error_indict=x_error_indict, x_true=x_true,
                       x_est=x_est, f=f)
            if s in spread_results:
                spread_results[s].append(r)
            else:
                spread_results[s] = [r]

    mpl.rcParams['legend.fontsize'] = 10

    colors = ['b','g','r','c','m','y','k']
    spread_order = sorted(spread_results.keys())

    # Plot a layer for each spread level
    fig = plt.figure()
    plt1 = fig.add_subplot(1,1,1,projection='3d')
    # plt2 = fig.add_subplot(1,2,2,projection='3d')
    for (k,c) in zip(spread_order,colors):
        v = spread_results[k]
        xs,ys,z1s,z2s = zip(*[(x.inertia,x.balancing,x.x_flow_error,
                               x.x_error_indict) for x in v])
        # ax = fig.gca(projection='3d')
        plot_results_wireframe(plt1,xs,ys,z1s, x_label='Hysteresis', y_label='Load balancing',
                     z_label='% flow error', plot_label='Interference = %s' % k, c=c)
        # plot_results_wireframe(plt2,xs,ys,z2s, x_label='Hysteresis', y_label='Load balancing',
        #                        z_label='% dist error', plot_label='Interference = %s' % k, c=c)
        plt.hold(True)
    plt.legend()
    plt.title('% Route flow error under handoff noise model')

    plt.show()
    ipdb.set_trace()

