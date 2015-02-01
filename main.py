import ipdb

import numpy as np
import scipy.io
import logging
from pprint import pprint

import config as c

from helpers import array, to_sp, to_np, parser, update_args, experiment_LS, \
    experiment_LSQR
from HighwayNetwork import HighwayNetwork

# dependencies from traffic-estimation-wardrop
from isttt2014_experiments import synthetic_data
import path_solver
import Waypoints as WP
from linkpath import LinkPath

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
    paths_sampled = sample_paths(paths, start_pos, end_pos,
                                 free_flow_travel_time, m=2)
    return paths_sampled

def generate_data_UE(data=None, export=False, SO=False, demand=3, m=20,
                          withODs=False, NLP=122):
    path='los_angeles_data_2.mat'
    graph, x_true, l, path_wps, wp_trajs, obs, wp = synthetic_data(data, SO, demand,
                                                    m=m, path=path, fast=False)
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

    return data, graph, wp_trajs

def scenario(params=None,m=2):
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
    data, graph, wp_trajs = generate_data_UE(data=config, SO=SO, NLP=args.NLP,
                                             m=m)

    if 'error' in data:
        return {'error' : data['error']}

    return args, data, graph, wp_trajs

def add_noise(data, paths_sampled, cps, num_cars=100, num_delays=10, m=2,
              tlimit=100,spreadlist=None, inertialist=None, balancinglist=None):
    HN = HighwayNetwork(data['cell_pos'], data['x_true'], paths_sampled)
    HN_data = HN.go(num_cars, num_delays, tlimit=tlimit, cellpaths=cps,
                            spread=spreadlist, inertia=inertialist,
                            balancing=balancinglist)
    return HN_data

def solve(args, data, noisy=False):
    eq = 'CP' if args.use_CP else 'OD'
    if args.solver == 'LS':
        output = experiment_LS(args, full=args.all_links, init=args.init,
                               L=args.use_L, OD=args.use_OD, CP=args.use_CP,
                               LP=args.use_LP, eq=eq, data=data, noisy=noisy)
    elif args.solver == 'LSQR':
        output = experiment_LSQR(args, full=args.all_links, L=args.use_L,
                                 OD=args.use_OD, CP=args.use_CP, LP=args.use_LP,
                                 data=data, noisy=noisy)

    if args.output == True:
        pprint(output)

    return output

def simplex(paths_sampled,cp_canonical):
    """Build simplex constraints from lp flows
    """
    from cvxopt import spmatrix
    n = len(cp_canonical)
    m = len(paths_sampled)
    if n == 0:
        return None, None
    I, J = [], []
    for i, path in enumerate(paths_sampled):
        I.append(cp_canonical.index(path))
        J.append(i)
    U = to_sp(spmatrix(1.0, I, J, (n, m)))
    return U

def construct_U(data,paths_sampled,cps):
    cp_list = []
    for path in paths_sampled:
        HN = HighwayNetwork(data['cell_pos'], np.array([1]), [path])
        HN_data = next(HN.go(1, 1, spread=[0], inertia=[0], balancing=[0]))
        cp = HN.getCellpaths()
        cp_list.append(cp)
    ipdb.set_trace()

    # We define this list of cp tuples to be the canonical one (including ordering)
    # This is the variable that will be issued to future calls of HN.go
    cp_canonical = list(set(cp_list))

    # Next we construct the U matrix based off of this canonical list
    U = simplex(paths_sampled,cp_canonical)

    return U, cp_canonical

def test_U(data, paths_sampled, cps, U, num_cars=100, num_delays=10, tlimit=100):
    HN = HighwayNetwork(data['cell_pos'], data['x_true'], paths_sampled)
    f,rest,(s,i,b) = next(HN.go(num_cars, num_delays, tlimit=tlimit, cellpaths=cps,
                    spread=[0], inertia=[0], balancing=[0]))
    x = data['x_true']
    logging.info('||Ux-f|| = %0.4f' % np.norm(U.dot(x)-f))

def experiment(m=2):
    args, data, graph, wp_trajs = scenario(m=20)
    # output_control = solve(args, data)
    # logging.info('Control flow error: %0.4f' % \
    #              output_control['percent flow allocated incorrectly'][-1])
    outputs = []
    f_orig = data['f']

    paths_sampled = generate_sampled_UE(graph,m=m)
    cps, cp_paths, cp_flow = zip(*wp_trajs)
    cps = [tuple(cpp) for cpp in cps]
    U, cp_canonical = construct_U(data,paths_sampled,cps)
    ipdb.set_trace()

    test_U(data, paths_sampled, cp_canonical, U)

    # spreadlist = (np.logspace(0,1,10, base=3)-1)/10
    # inertialist = (np.logspace(0,1,10, base=3)-1)/10
    # balancinglist = (np.logspace(0,1,10, base=3)-1)/10
    num_cars = 1000
    num_delays = 10
    tlimit = 100
    spreadlist = [0, 0.02,0.05]
    inertialist = [0, 0.02,0.05]
    balancinglist = [0, 0.002,0.005]

    HN_data = add_noise(data, paths_sampled, cp_canonical, num_cars=num_cars, m=m,
                          num_delays=num_delays, tlimit=tlimit,
                          spreadlist=spreadlist, inertialist=inertialist,
                          balancinglist=balancinglist)

    for f,rest,(s,i,b) in HN_data:
        # Replace f with new noisy f
        ipdb.set_trace()

        data['f'] = array(f)

        output = solve(args, data, noisy=True)
        outputs.append(output)
        logging.info('Params: (%0.3f,%0.3f,%0.3f)' % (s,i,b))
        logging.info('Flow error: %0.4f' % \
                 output['percent flow allocated incorrectly'][-1])
        logging.info('Rest: %0.4f' % rest)
    ipdb.set_trace()
    # return output_control, outputs
    return outputs

def plot_results(output_control, outputs):
    pass

if __name__ == "__main__":
    # scenario()
    import sys, random
    # myseed = random.randint(0, sys.maxint)
    myseed = 4294967295
    print "Random seed:", myseed
    np.random.seed(myseed)
    random.seed(myseed)
    experiment(m=10)
