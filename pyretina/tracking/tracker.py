class Tracker(object):
  def set_event(self, event):
    pass

  def search(self):
    pass

class RetinaTracker(Tracker):
  def __init__(self, retina_model, optimizer):
    self._retina_model = retina_model
    self._optimizer = optimizer

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def retina_model(self):
    return self._retina_model

  def set_event(self, event):
    self._retina_model.set_event(event)

  def fit_stream(self, event_stream):
    pass

  def fit(self, *args, **kwargs):
    raise NotImplementedError()

  def predict(self, event):
    pass
