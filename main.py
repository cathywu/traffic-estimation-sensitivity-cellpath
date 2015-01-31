import ipdb

import numpy as np
import scipy.io
import logging

import config as c

from helpers import array, to_sp, parser, update_args
from isttt2014_experiments import synthetic_data
import path_solver
import Waypoints as WP
from linkpath import LinkPath
from HighwayNetwork import HighwayNetwork

def sample_paths(paths, start_pos, end_pos, free_flow_travel_time, m=2):
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
    return paths_sampled

def generate_sampled_UE(g,m=2):
    """
    Generate sampled paths based on free flow travel time
    :param g:
    :param wp:
    :return:
    """
    paths = g.get_paths()
    start_pos, end_pos = g.get_links()
    free_flow_travel_time = array(g.get_ffdelays())
    paths_sampled = sample_paths(paths, start_pos, end_pos, free_flow_travel_time, m=2)
    return paths_sampled

def generate_data_UE(data=None, export=False, SO=False, demand=3, N=30,
                          withODs=False, NLP=122):
    path='los_angeles_data_2.mat'
    graph, x_true, l, path_wps, wp_trajs, obs, wp = synthetic_data(data, SO, demand,
                                                    N, path=path, fast=False)
    x_true = array(x_true)
    obs=obs[0]
    A_full = path_solver.linkpath_incidence(graph)
    A = to_sp(A_full[obs,:])
    A_full = to_sp(A_full)
    U,f = WP.simplex(graph, wp_trajs, withODs)
    T,d = path_solver.simplex(graph)

    # cell locations
    cell_pos = array(wp.wp.values())

    data = {'A_full': A_full, 'b_full': A_full.dot(x_true),
            'A': A, 'b': A.dot(x_true), 'x_true': x_true,
            'T': to_sp(T), 'd': array(d), 'U': to_sp(U), 'f': array(f),
            'cell_pos' : cell_pos}

    if NLP is not None:
        lp = LinkPath(graph,x_true,N=NLP)
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

    return data, graph

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
    data, graph = generate_data_UE(data=config, SO=SO, NLP=args.NLP)
    paths_sampled = generate_sampled_UE(graph,m=2)
    HN = HighwayNetwork(data['cell_pos'], data['x_true'], paths_sampled)
    HN.go(100, 10, tlimit=100)
    # Print results
    for k, v in HN.paths.iteritems():
        print k, ":", v
    print len(HN.paths)

    if 'error' in data:
        return {'error' : data['error']}

    ipdb.set_trace()

if __name__ == "__main__":
    scenario()
