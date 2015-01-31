ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

DATA_DIR = '/Users/cathywu/Dropbox/PhD/traffic-estimation-sensitivity-cellpath/data'

REPOSITORIES = { \
    'LS': '/Users/cathywu/Dropbox/PhD/traffic-estimation',
    'wardrop': '/Users/cathywu/Dropbox/PhD/traffic-estimation-wardrop', }

import os
for (k,v) in REPOSITORIES.iteritems():
    # add to path after current dir
    os.sys.path.insert(1,v)
