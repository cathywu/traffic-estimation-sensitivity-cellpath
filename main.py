import ipdb

import pickle
import numpy as np
import numpy.linalg as la
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
    graph, x_true, l, path_wps, wp_trajs, obs, wp = synthetic_data(data, SO,
                                            demand, m=m, path=path, fast=False)
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
    data, graph = generate_data_UE(data=config, SO=SO, NLP=args.NLP,m=m)

    if 'error' in data:
        return {'error' : data['error']}

    return args, data, graph

def solve(args, data, noisy=False, save_x=False):
    eq = 'CP' if args.use_CP else 'OD'
    if args.solver == 'LS':
        output = experiment_LS(args, full=args.all_links, init=args.init,
                               L=args.use_L, OD=args.use_OD, CP=args.use_CP,
                               LP=args.use_LP, eq=eq, data=data, noisy=noisy,
                               save_x=save_x)
    elif args.solver == 'LSQR':
        output = experiment_LSQR(args, full=args.all_links, L=args.use_L,
                                 OD=args.use_OD, CP=args.use_CP, LP=args.use_LP,
                                 data=data, noisy=noisy)

    if args.output == True:
        pprint(output)

    return output

def simplex(cp_list,cp_canonical):
    """Build simplex constraints from lp flows
    """
    from cvxopt import spmatrix
    n = len(cp_canonical)
    m = len(cp_list)
    if n == 0:
        return None, None
    I, J = [], []
    for i, path in enumerate(cp_list):
        I.append(cp_canonical.index(path))
        J.append(i)
    U = to_sp(spmatrix(1.0, I, J, (n, m)))
    return U

def construct_U(data,paths_sampled):
    cp_list = []
    for path in paths_sampled:
        HN = HighwayNetwork(data['cell_pos'], [1], [path])
        HN_data = next(HN.go(2, 1, spread=[0], inertia=[0], balancing=[0]))
        cp = HN.getCellpaths()[0]
        cp_list.append(cp)

    # We define this list of cp tuples to be the canonical one (including ordering)
    # This is the variable that will be issued to future calls of HN.go
    cp_canonical = list(set(cp_list))

    # Next we construct the U matrix based off of this canonical list
    U = simplex(cp_list,cp_canonical)

    return U, cp_canonical

def test_U(data, paths_sampled, cps, U, num_cars=1000, num_delays=10):
    HN = HighwayNetwork(data['cell_pos'], data['x_true'], paths_sampled)
    f,rest,(s,i,b) = next(HN.go(num_cars, num_delays, cellpaths=cps,
                    spread=[0], inertia=[0], balancing=[0]))
    x_true = HN.x_true
    # x = data['x_true']
    logging.info('Noiseless ||Ux-f|| = %0.4f' % la.norm(U.dot(x_true)-f))
    return f

def experiment(m=2,fname=None):
    if fname is None:
        args, data, graph = scenario(m=20)
        paths_sampled = generate_sampled_UE(graph,m=m)
        U, cp_canonical = construct_U(data,paths_sampled)
        data['U'] = U

        f_true = test_U(data, paths_sampled, cp_canonical, U)
        with open('%s/temp.pkl' % c.DATA_DIR, 'w') as out:
            pickle.dump((args, data, paths_sampled, cp_canonical, f_true), out)
    else:
        with open('%s/%s' % (c.DATA_DIR,fname)) as fin:
            (args, data, paths_sampled, cp_canonical, f_true) = pickle.load(fin)
        if args.log in c.ACCEPTED_LOG_LEVELS:
            logging.basicConfig(level=eval('logging.'+args.log))
        U,f,x = data['U'], f_true, data['x_true']
        logging.info('Init, ||Ux-f|| = %0.4f' % la.norm(U.dot(x)-f))

    # spreadlist = (np.logspace(0,1,10, base=3)-1)/10
    # inertialist = (np.logspace(0,1,10, base=3)-1)/10
    # balancinglist = (np.logspace(0,1,10, base=3)-1)/10
    num_cars = 1000
    num_delays = 10
    spreadlist = [0, 0.05, 0.1]
    inertialist = [0, 0.05, 0.1, 0.2, 0.3]
    balancinglist = [0, .0004, .0008, .0012, .002] # [0, 0.0005, 0.001, 0.002, 0.003]

    # spreadlist = [0]
    # inertialist = [0]
    # balancinglist = [0, 0.001, 0.002, 0.003, 0.004]

    outputs = []
    f_orig = data['f']
    # output_control = solve(args, data)
    # logging.info('Control flow error: %0.4f' % \
    #              output_control['percent flow allocated incorrectly'][-1])

    HN = HighwayNetwork(data['cell_pos'], data['x_true'], paths_sampled)
    HN_data = HN.go(num_cars, num_delays, cellpaths=cp_canonical,
                    spread=spreadlist, inertia=inertialist,
                    balancing=balancinglist)
    # Noiseless-ify
    data['x_true'] = HN.x_true
    data['b'] = data['A'].dot(data['x_true'])
    data['d'] = data['T'].dot(data['x_true'])

    delimiter = "|"
    captions = ['spread', 'inertia', 'balancing', 'x flow error', 'x total',
                'rest', 'Ux-f', 'x flow error in dict', 'x_true', 'x_est', 'f']
    formats = ['%0.8f', '%0.8f', '%0.8f', '%0.8f', '%0.8f', '%0.8f', '%0.8f',
               '%0.8f', '%s', '%s', '%s']
    format_string = delimiter.join(formats)

    with open('%s/%s' % (c.DATA_DIR,'results.txt'), 'w') as fres:
        fres.write('%s\n' % delimiter.join(captions))
        for f,rest,(s,i,b) in HN_data:
            # Replace f with new noisy f
            data['f'] = array(f)
            # ipdb.set_trace()
            output = solve(args, data, noisy=True, save_x=True)
            outputs.append((output,f,rest,(s,i,b)))
            x_error = output['percent flow allocated incorrectly'][-1]
            logging.info('Params: (%0.3f,%0.3f,%0.3f)' % (s,i,b))
            logging.info('Flow error: %0.4f' % x_error)
            logging.info('Rest: %0.4f' % rest)
            U,f,x = data['U'], data['f'], data['x_true']
            Uxf = la.norm(U.dot(x)-f)
            x_total = np.sum(data['x_true'])

            # scaled % route flow error
            x_est = output['x_est']
            x_true = output['x_true']
            indict = 1 - rest / x_total
            x_error_indict = np.sum(indict * x_true - x_est) / np.sum(indict * x_true)

            write_out = format_string % (s, i, b, x_error, x_total, rest, Uxf,
                                         x_error_indict,
                                         np.array_str(x_true,max_line_width=np.Inf),
                                         np.array_str(x_est,max_line_width=np.Inf),
                                         np.array_str(f,max_line_width=np.Inf))
            fres.write('%s\n' % write_out)
            ipdb.set_trace()

    ipdb.set_trace()
    # return output_control, outputs
    return outputs

def plot_results(output_control, outputs):
    pass

if __name__ == "__main__":
    # scenario()
    import sys, random
    # myseed = random.randint(0, sys.maxint)
    myseed = 4253648295
    print "Random seed:", myseed
    np.random.seed(myseed)
    random.seed(myseed)
    experiment(m=10,fname='temp.pkl')
    # experiment(m=10)
