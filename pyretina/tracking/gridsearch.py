from tracker import RetinaTracker
from pyretina.optimize import GridOptimizer
from pyretina.retina import ReferencePlaneRetinaModel

class GridSearchTracker(RetinaTracker):
  def __init__(self, ranges, threshold = 2.9, filter_size = 3, reference_z = 700.0):
    retina_model = ReferencePlaneRetinaModel(reference_z=reference_z)
    optimizer = GridOptimizer(retina_model, ranges, threshold=threshold, filter_size=filter_size)
    
    super(GridSearchTracker, self).__init__(retina_model, optimizer)

  def fit(self, events):
    pass

  def predict(self, event):
    self.set_event(event.hits)
    return self.optimizer.maxima()