from scipy.sparse import csr_matrix
import argparse
import numpy as np
import time
import logging
import numpy.linalg as la

import config as c

from python.util import load_data, solver_input
from python import util
from python.c_extensions.simplex_projection import simplex_projection
from python import BB, LBFGS, DORE, solvers

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

def experiment_LS(args, test=None, data=None, full=True, L=True, OD=True,
                  CP=True, LP=True, eq='CP', init=True):
    """
    Least squares experiment
    :param test:
    :return:
    """
    ## LS experiment
    ## TODO: invoke solver
    init_time = time.time()
    if data is None and test is not None:
        fname = '%s/%s' % (c.DATA_DIR,test)
        A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
            load_data(fname, full=full, L=L, OD=OD, CP=CP, LP=LP, eq=eq,
                      init=init)
    else:
        A, b, N, block_sizes, x_true, nz, flow, rsort_index, x0, out = \
            solver_input(data, full=full, L=L, OD=OD, CP=CP, LP=LP,
                         eq=eq, init=init)
    init_time = time.time() - init_time
    output = out
    output['init_time'] = init_time

    # x0 = np.array(util.block_e(block_sizes - 1, block_sizes))

    if args.noise:
        b_true = b
        delta = np.random.normal(scale=b*args.noise)
        b = b + delta

    if block_sizes is not None:
        logging.debug("Blocks: %s" % block_sizes.shape)
    # z0 = np.zeros(N.shape[1])
    if N is None or (block_sizes-1).any() == False:
        iters, times, states = [0],[0],[x0]
        x_last, error, output = LS_postprocess(states,x0,A,b,x_true,scaling=flow,
                                               block_sizes=block_sizes,N=N,
                                               output=output,is_x=True)
    else:
        iters, times, states = LS_solve(A,b,x0,N,block_sizes,args)
        x_last, error, output = LS_postprocess(states,x0,A,b,x_true,scaling=flow,
                                               block_sizes=block_sizes,N=N,
                                               output=output)

    # LS_plot(x_last, times, error)
    output['duration'] = np.sum(times)

    output['iters'], output['times'] = list(iters), list(times)
    return output

def LS_solve(A,b,x0,N,block_sizes,args):
    z0 = util.x2z(x0,block_sizes)
    target = A.dot(x0)-b

    options = { 'max_iter': 300000,
                'verbose': 1,
                'opt_tol' : 1e-30,
                'suff_dec': 0.003, # FIXME unused
                'corrections': 500 } # FIXME unused
    AT = A.T.tocsr()
    NT = N.T.tocsr()

    f = lambda z: 0.5 * la.norm(A.dot(N.dot(z)) + target)**2
    nabla_f = lambda z: NT.dot(AT.dot(A.dot(N.dot(z)) + target))

    def proj(x):
        projected_value = simplex_projection(block_sizes - 1,x)
        # projected_value = pysimplex_projection(block_sizes - 1,x)
        return projected_value

    import time
    iters, times, states = [], [], []
    def log(iter_,state,duration):
        iters.append(iter_)
        times.append(duration)
        states.append(state)
        start = time.time()
        return start

    logging.debug('Starting %s solver...' % args.method)
    if args.method == 'LBFGS':
        LBFGS.solve(z0+1, f, nabla_f, solvers.stopping, log=log,proj=proj,
                    options=options)
        logging.debug("Took %s time" % str(np.sum(times)))
    elif args.method == 'BB':
        BB.solve(z0,f,nabla_f,solvers.stopping,log=log,proj=proj,
                 options=options)
    elif args.method == 'DORE':
        # setup for DORE
        alpha = 0.99
        lsv = util.lsv_operator(A, N)
        logging.info("Largest singular value: %s" % lsv)
        A_dore = A*alpha/lsv
        target_dore = target*alpha/lsv

        DORE.solve(z0, lambda z: A_dore.dot(N.dot(z)),
                   lambda b: N.T.dot(A_dore.T.dot(b)), target_dore, proj=proj,
                   log=log,options=options,record_every=100)
        A_dore = None
    logging.debug('Stopping %s solver...' % args.method)
    return iters, times, states

def LS_postprocess(states, x0, A, b, x_true, scaling=None, block_sizes=None,
                   output=None, N=None, is_x=False):
    if scaling is None:
        scaling = np.ones(x_true.shape)
    if output is None:
        output = {}
    d = len(states)
    if not is_x:
        x_hat = N.dot(np.array(states).T) + np.tile(x0,(d,1)).T
    else:
        x_hat = np.array(states).T
    x_last = x_hat[:,-1]
    n = x_hat.shape[1]

    logging.debug("Shape of x0: %s" % repr(x0.shape))
    logging.debug("Shape of x_hat: %s" % repr(x_hat.shape))
    logging.debug('A: %s' % repr(A.shape))
    if block_sizes is not None:
        logging.debug('blocks: %s' % repr(block_sizes.shape))
    output['AA'] = A.shape
    output['blocks'] = block_sizes.shape if block_sizes is not None else None

    # Objective error, i.e. 0.5||Ax-b||_2^2
    starting_error = 0.5 * la.norm(A.dot(x0)-b)**2
    opt_error = 0.5 * la.norm(A.dot(x_true)-b)**2
    diff = A.dot(x_hat) - np.tile(b,(d,1)).T
    error = 0.5 * np.diag(diff.T.dot(diff))
    output['0.5norm(Ax-b)^2'], output['0.5norm(Ax_init-b)^2'] = error, starting_error
    logging.debug('0.5norm(Ax-b)^2: %8.5e\n0.5norm(Ax_init-b)^2: %8.5e' % \
                  (error[-1], starting_error))
    output['0.5norm(Ax*-b)^2'] = opt_error
    logging.debug('0.5norm(Ax*-b)^2: %8.5e' % opt_error)

    # Route flow error, i.e ||x-x*||_1
    x_true_block = np.tile(x_true,(n,1))
    x_diff = x_true_block-x_hat.T

    scaling_block = np.tile(scaling,(n,1))
    x_diff_scaled = scaling_block * x_diff
    x_true_scaled = scaling_block * x_true_block

    # most incorrect entry (route flow)
    dist_from_true = np.max(x_diff_scaled,axis=1)
    logging.debug('max|f * (x-x_true)|: %.5f' % dist_from_true[-1])
    output['max|f * (x-x_true)|'] = dist_from_true

    # num incorrect entries
    wrong = np.bincount(np.where(x_diff > 1e-3)[0])
    output['incorrect x entries'] = wrong
    logging.debug('incorrect x entries: %s' % wrong[-1] if wrong.size > 0 else 'NA')

    # % route flow error
    per_flow = np.sum(np.abs(x_diff_scaled), axis=1) / np.sum(x_true_scaled, axis=1)
    output['percent flow allocated incorrectly'] = per_flow
    logging.debug('percent flow allocated incorrectly: %f' % per_flow[-1] if \
                      per_flow.size > 0 else 'NA')

    # initial route flow error
    start_dist_from_true = np.max(scaling * np.abs(x_true-x0))
    logging.debug('max|f * (x_init-x_true)|: %.5f' % start_dist_from_true)
    output['max|f * (x_init-x_true)|'] = start_dist_from_true

    return x_last, error, output

def experiment_LSQR(args, test=None, data=None, full=False, L=True, OD=True,
                    CP=True, LP=True, damp=0.0):
    init_time = time.time()
    A, b, x0, x_true, out = solver_input(data, full=full, L=L, OD=OD, CP=CP,
                                         LP=LP, solve=True, damp=damp)
    init_time = time.time() - init_time
    output = out
    output['init_time'] = init_time

    if A is None:
        output['error'] = "Empty objective"
        return output

    _, _, output = LS_postprocess([x0],x0,A,b,x_true,output=output,is_x=True)

    output['duration'], output['iters'], output['times'] = 0, [0], [0]
    return output
