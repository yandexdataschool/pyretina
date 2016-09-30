from tracker import RetinaTracker
from pyretina.optimize import GD
from pyretina.retina import ReferencePlaneRetinaModel

class GDTracker(RetinaTracker):
  def __init__(self, seeder_model, n_seeds, n_steps,
               alpha_regime, sigma_regime,
               threshold=2.9, reference_z = 100.0):

    retina_model = ReferencePlaneRetinaModel(reference_z=reference_z)
    optimizer = GD(retina_model, seeder_model,
                   n_seeds=n_seeds, n_steps=n_steps,
                   alpha_regime=alpha_regime, sigma_regime=sigma_regime,
                   threshold=threshold)

    super(GDTracker, self).__init__(retina_model, optimizer)

  def fit(self, events):
    pass

  def predict(self, event):
    self.set_event(event.hits)
    return self.optimizer.maxima()