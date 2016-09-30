from tracker import RetinaTracker
from optimize import GradBased
from retina import ReferencePlaneRetinaModel

class NNTracker(RetinaTracker):
  def __init__(self, seeder_model, n_seeds, n_steps,
               normalization_coefs, alpha = 1.0,
               threshold=2.9, reference_z = 100.0):

    retina_model = ReferencePlaneRetinaModel(reference_z=reference_z)
    optimizer = GradBased(
      retina_model, seeder_model,
      n_seeds=n_seeds, n_steps=n_steps,
      normalization_coefs=normalization_coefs, alpha=alpha,
      n_units=250,
      threshold=threshold
    )

    super(NNTracker, self).__init__(retina_model, optimizer)

  def fit(self, events, sigma_train, learning_rate):
    loss = []
    for event in events:
      l = self.optimizer.train(event, sigma_train, learning_rate)
      loss.append(l)

    return loss

  def predict(self, event):
    self.set_event(event.hits)
    return self.optimizer.maxima()