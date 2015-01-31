import ipdb

import numpy as np
from scipy.sparse import csr_matrix
import scipy.io
import argparse
import logging

import config as c

from isttt2014_experiments import synthetic_data
import path_solver
import Waypoints as WP
from linkpath import LinkPath

# Clean array wrapper
def array(x):
    return np.atleast_1d(np.squeeze(np.array(x)))

def to_sp(X):
    if X is None:
        return None
    return csr_matrix((array(X.V),(array(X.I),array(X.J))), shape=X.size)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='log', nargs='?', const='INFO',
                        default='WARN', help='Set log level (default: WARN)')
    parser.add_argument('--output',dest='output',action='store_true',
                        default=False,help='Print output toggle (false)')

    parser.add_argument('--solver',dest='solver',type=str,default='LS',
                        help='Solver name') # CS/BI/LS
    parser.add_argument('--model',dest='model',type=str,default='P',
                        help='Macro traffic dynamics model') # P/UE/SO
    parser.add_argument('--noise',dest='noise',type=float,default=None,
                        help='Noise level')
    parser.add_argument('--all_links',dest='all_links',action='store_true',
                        default=False,help='All links observed (false)')

    # Sensor toggles
    parser.add_argument('--use_L',dest='use_L',action='store_false',
                        default=True,help='Use L sensors (true)')
    parser.add_argument('--use_OD',dest='use_OD',action='store_false',
                        default=True,help='Use OD sensors (true)')
    parser.add_argument('--use_CP',dest='use_CP',action='store_false',
                        default=True,help='Use CP sensors (true)')
    parser.add_argument('--use_LP',dest='use_LP',action='store_false',
                        default=True,help='Use LP sensors (true)')

    # LS solver only
    parser.add_argument('--method',dest='method',type=str,default='BB',
                        help='LS only: Least squares method')
    parser.add_argument('--init',dest='init',action='store_true',
                        default=False,help='LS only: initial solution from data')

    # LSQR solver only
    parser.add_argument('--damp',dest='damp',type=float,default=0.0,
                        help='LSQR only: damping factor')

    # Sparsity
    parser.add_argument('--sparse',dest='sparse',action='store_true',
                        default=False,help='BI/P only: Sparse toggle for route flow sampling')

    # Linkpath and cellpath sensing
    parser.add_argument('--NLP',dest='NLP',type=int,default=9,
                        help='Number of linkpath sensors (sampled uniformly)')
    parser.add_argument('--NB',dest='NB',type=int,default=48,
                        help='Number of cells sampled uniformly in bbox')
    parser.add_argument('--NS',dest='NS',type=int,default=0,
                        help='Number of cells sampled uniformly in region')
    parser.add_argument('--NL',dest='NL',type=int,default=9,
                        help='Number of cells sampled uniformly in links')

    # P model only
    parser.add_argument('--nrow',dest='nrow',type=int,default=3,
                        help='P only: Number of rows in grid network')
    parser.add_argument('--ncol',dest='ncol',type=int,default=4,
                        help='P only: Number of rows in grid network')
    parser.add_argument('--nodroutes',dest='nodroutes',type=int,default=15,
                        help='P only: Number of routes per OD pair')
    return parser

def update_args(args, params):
    """
    Update argparse object with attributes from params dictionary
    :param args:
    :param params:
    :return:
    """
    args.solver = params['solver'] # LS/BI/CS
    args.model = params['model'] # P/UE/SO
    args.sparse = bool(params['sparse']) # sparse toggle for route flow sampling
    if 'all_links' in params:
        args.all_links = bool(params['all_links'])

    # Sensor toggles
    args.use_L = bool(params['use_L']) if 'use_L' in params else True
    args.use_OD = bool(params['use_OD']) if 'use_OD' in params else True
    args.use_CP = bool(params['use_CP']) if 'use_CP' in params else True
    args.use_LP = bool(params['use_LP']) if 'use_LP' in params else True

    # Sensor configurations
    args.NLP = int(params['NLP']) # number of linkpath sensors (sampled randomly)
    args.NB = int(params['NB'])   # number of cells sampled uniformly in bbox
    args.NS = int(params['NS'])   # number of cells sampled uniformly in region
    args.NL = int(params['NL'])   # number of cells sampled uniformly in links

    if args.model == 'P':
        # For --model P only:
        args.nrow = int(params['nrow']) # number of rows in grid network
        args.ncol = int(params['ncol']) # number of cols in grid network
        args.nodroutes = int(params['nodroutes']) # number of routes per OD pair

    if args.solver == 'LS':
        # For --solver LS only:
        args.method = params['method'] # BB/LBFGS/DORE
        args.init = bool(params['init']) # True/False

    if args.solver == 'LSQR':
        args.damp = float(params['damp'])

    return args

def generate_sampled_data(data=None, export=False, SO=False, trials=10, demand=3, N=30,
                     plot=False, withODs=False, NLP=122):
    path='los_angeles_data_2.mat'
    g, x_true, l, path_wps, wp_trajs, obs = synthetic_data(data, SO, demand, N,
                                                           path=path,fast=False)
    x_true = array(x_true)
    obs=obs[0]
    A_full = path_solver.linkpath_incidence(g)
    A = to_sp(A_full[obs,:])
    A_full = to_sp(A_full)
    U,f = WP.simplex(g, wp_trajs, withODs)
    T,d = path_solver.simplex(g)

    # FOR SENSITIVITY
    # paths, each as a sequence of link ids
    paths_obj = [[(v.startnode,v.endnode,1) for v in p.links] for (k,p) in g.paths.iteritems()]
    paths = [[g.indlinks[v] for v in p] for p in paths_obj]

    # x and y position for each link
    link_pos,ind= zip(*sorted([((g.nodes[a].position,g.nodes[b].position),v) \
                               for ((a,b,c),v) in g.indlinks.iteritems()],
                              key=lambda x: x[1]))
    start_pos, end_pos = zip(*link_pos)
    start_pos, end_pos = np.array(start_pos), np.array(end_pos)

    # free flow travel time for each link
    free_flow_travel_time = array(g.get_ffdelays())

    # sampled paths
    m = 2
    paths_sampled = []
    for path in paths:
        path_sampled = []
        for link in path:
            x_start, y_start = start_pos[link]
            x_end, y_end = end_pos[link]
            tt = free_flow_travel_time[link]
            interp_x = np.linspace(x_start,x_end,num=tt*m)
            interp_y = np.linspace(y_start,y_end,num=tt*m)
            if len(interp_y) <= 1:
                path_sampled.extend(zip(interp_x,interp_y))
            else:
                path_sampled.extend(zip(interp_x,interp_y)[:-1])
        paths_sampled.append(path_sampled)

    data = {'A_full': A_full, 'b_full': A_full.dot(x_true),
            'A': A, 'b': A.dot(x_true), 'x_true': x_true,
            'T': to_sp(T), 'd': array(d),
            'U': to_sp(U), 'f': array(f),
            'paths' : paths, 'link_start': start_pos, 'link_end': end_pos,
            'link_travel_time': free_flow_travel_time,
            'paths_sampled' : paths_sampled}

    if NLP is not None:
        lp = LinkPath(g,x_true,N=NLP)
        lp.update_lp_flows()
        V,g = lp.simplex_lp()
        data['V'], data['g'] = V, g

    # Export
    if export:
        if not SO:
            fname = '%s/UE_graph.mat' % c.DATA_DIR
        else:
            fname = '%s/SO_graph.mat' % c.DATA_DIR
        scipy.io.savemat(fname, data, oned_as='column')

    return data

def scenario(params=None, log='INFO'):
    # use argparse object as default template
    p = parser()
    args = p.parse_args()
    if args.log in c.ACCEPTED_LOG_LEVELS:
        logging.basicConfig(level=eval('logging.'+args.log))
    if params is not None:
        args = update_args(args, params)

    print args

    SO = True if args.model == 'SO' else False
    # N0, N1, scale, regions, res, margin
    config = (args.NB, args.NL, 0.2, [((3.5, 0.5, 6.5, 3.0), args.NS)], (6,3), 2.0)
    # data[0] = (20, 40, 0.2, [((3.5, 0.5, 6.5, 3.0), 20)], (12,6), 2.0)
    # data[1] = (10, 20, 0.2, [((3.5, 0.5, 6.5, 3.0), 10)], (10,5), 2.0)
    # data[2] = (5, 10, 0.2, [((3.5, 0.5, 6.5, 3.0), 5)], (6,3), 2.0)
    # data[3] = (3, 5, 0.2, [((3.5, 0.5, 6.5, 3.0), 2)], (4,2), 2.0)
    # data[4] = (1, 3, 0.2, [((3.5, 0.5, 6.5, 3.0), 1)], (2,2), 2.0)
    # TODO trials?
    data = generate_sampled_data(data=config, SO=SO, NLP=args.NLP)
    if 'error' in data:
        return {'error' : data['error']}

    ipdb.set_trace()

if __name__ == "__main__":
    scenario()
