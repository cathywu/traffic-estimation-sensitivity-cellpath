from random import shuffle, random, uniform, randint, seed
from math import log10

def uniq(seq):
    last = [None]
    setlast = lambda x: last.append(x)
    return tuple([ x for x in seq if not (x is None or x == last[-1] or setlast(x))])

def weighted_choice(weights):
   total = sum(weights)
   r = uniform(0, total)
   upto = 0
   for i, w in enumerate(weights):
      upto += w
      if upto > r:
         return i
   assert False, "Shouldn't get here"

class HighwayNetwork:

  def __init__(self, cellPositions, flows, routes, numCars, numDelays, tlimit=None, spread=0, inertia=0, balancing=0):
    # spread : random network interference degrading signal strength
    self.spread = spread
    # inertia : preference for an individual car to stay on the same tower
    self.inertia = inertia
    # balancing : disincentivize more heavily loaded towers
    self.balancing = balancing

    self.cellPositions = cellPositions
    cars = []
    '''
    for (flow, route) in zip(flows, routes):
      # Add a N = flow new cars travelling on each respective route
      for i in xrange(flow):
        # TODO: disperse the cars in time by adding a random number of None's at the beginning of the route
        cars.append(route)
    '''
    print "Deploying cars..."
    for d in xrange(numDelays):
      # add numCars every timestep for numDelay timesteps
      delay = [None] * d
      for c in xrange(numCars):
        # pick a route for each car according to flows
        route = weighted_choice(flows)
        cars.append(delay + routes[route])
    shuffle(cars)
    # transpose to get where each car is at each timestep
    self.timesteps = map(list, map(None, *cars))
    if tlimit:
      self.timesteps = self.timesteps[:tlimit]

    print "Calculating signal strength..."
    self.buildRSSI()
    print "Determining tower assignment..."
    self.assignTowers()
    print "Done."
    self.collect()

  def calculateRSSI(self, car, tower):
    # spread : random network interference degrading signal strength
    d2 = (car[0] - tower[0])**2 + (car[1] - tower[1])**2
    fade = 10*log10(d2) + self.spread*random()
    return -fade

  def buildRSSI(self):
    for ts in self.timesteps:
      for i in xrange(len(ts)):
        if ts[i] is None:
          continue
        rssis = []
        for cell in self.cellPositions:
          rssis.append(self.calculateRSSI(ts[i], cell))
        ts[i] = rssis

  def assignTowers(self):
    for i, ts in enumerate(self.timesteps):
      towerLoad = [0] * len(self.cellPositions)
      for c, car in enumerate(ts):
        # inertia : stay on the same tower
        if car is None:
          continue
        if i > 0:
          lastTower = self.timesteps[i-1][c]
          if lastTower is not None:
            car[lastTower] += self.inertia

        # balancing : penalize heavily loaded towers
        for t, load in enumerate(towerLoad):
          car[t] -= self.balancing * load

        # pick winner
        tower = car.index(max(car))
        self.timesteps[i][c] = tower
        towerLoad[tower] += 1

  def collect(self):
    cars = map(uniq, map(None, *self.timesteps))
    paths = {}
    for car in cars:
      paths[car] = 1 + paths.get(car, 0)
    self.paths = paths
   
  def toString(self):
    for ts in self.timesteps:
      for car in ts:
        print car
      print
    print "~~~"

if __name__ == "__main__": 
  import sys
  myseed = randint(0, sys.maxint)
  print "Random seed:", myseed
  seed(myseed)

  simple = False

  """
  Set up environment
  """
  if simple:
    cellPositions = [ (1,1), (1,3), (3,1) ]
    flows = [ 1, 2, 3, 4 ]
    routes = [ [ (t, 2*t) for t in range(3) ], 
               [ (2*t, t) for t in range(3) ], 
               [ (t, 3-t) for t in range(4) ], 
               [ (2*t, 2*t) for t in range(3) ], 
             ]
  else:
    numCellTowers = 80

    import scipy.io
    data = scipy.io.loadmat('sensitivity_sample.mat')
    flows = [1 for x in data['x_true']]
    routes = [map(tuple, x[0]) for x in data['paths_sampled']]

    cellPositions = []
    x = [s[0] for r in routes for s in r]
    y = [s[1] for r in routes for s in r]
    maxx = max(x)
    maxy = max(y)
    minx = min(x)
    miny = min(y)
    for i in xrange(numCellTowers):
      cellPositions.append((minx + random() * (maxx - minx), miny + random() * (maxy - miny)))

  """
  Simulation
  """
  # Run simulation
  n = HighwayNetwork(cellPositions, flows, routes, 200, 10, tlimit = 100)
  # Print results
  for k, v in n.paths.iteritems():
    print k, ":", v
  print len(n.paths)
