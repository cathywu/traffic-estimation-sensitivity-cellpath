
from scipy.sparse import csr_matrix
import argparse
import numpy as np

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

