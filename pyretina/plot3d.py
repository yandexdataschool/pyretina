import numpy as np

from evaluate import binary_metrics

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_retina_results(predicted, test, thetas, phis, response, max_angle):
  m, predicted_mapping, test_mapping = binary_metrics(predicted, test, max_angle=max_angle)
  recognized = predicted_mapping == 1
  test_recognized = test_mapping == 1
  ghost = predicted_mapping == 0
  unrecognized = test_mapping == 0

  plt.figure()
  plt.contourf(thetas, phis, response, 20, cmap=cm.gist_gray)
  plt.colorbar()

  plt.scatter(predicted[recognized, 0], predicted[recognized, 1], color="green", marker="+",
              label="Recognized (%d)" % np.sum(test_recognized), s=80)

  plt.scatter(test[test_recognized, 0], test[test_recognized, 1], color="green", marker="o",
              s=40)


  plt.scatter(predicted[ghost, 0], predicted[ghost, 1], color="red", marker="x",
              label="Ghost (%d)" % np.sum(ghost), s=80)

  plt.scatter(test[unrecognized, 0], test[unrecognized, 1], color="red", marker="o",
              label="Unrecognized (%d)" % np.sum(unrecognized), s=80)

  #plt.legend()
  return plt

def plot_event_mayavi(event, tracks = None):
  from mayavi import mlab

  mlab.figure(bgcolor=(1, 1, 1))
  mlab.points3d(event[:, 0], event[:, 1], event[:, 2], color=(0.9, 0, 0.2), opacity=1.0, scale_factor=0.1)
  if tracks is not None:
    n_tracks = tracks.shape[0]
    tracks = np.vstack([
      [0, 0, 0],
      tracks
    ])

    connections = np.vstack([np.zeros(n_tracks + 1, dtype="int64"), np.arange(n_tracks + 1)]).T

    points = mlab.points3d(
      tracks[:, 0], tracks[:, 1], tracks[:, 2],
      scale_mode='none',
      scale_factor=0.03,
      color=(0, 0, 1)
    )

    points.mlab_source.dataset.lines = connections
    points.mlab_source.update()
    mlab.pipeline.surface(
      points, color=(0, 0, 1),
      representation='wireframe',
      line_width=2,
      opacity=0.5,
      name='Connections'
    )

  mlab.show()

def mininal_sq(ps):
  maxs = np.max(ps, axis=0)
  mins = np.min(ps, axis=0)
  delta = np.max(maxs - mins) / 2

  centers = (maxs + mins) / 2
  b = centers - delta
  t = centers + delta

  return [ [b[i], t[i]] for i in range(b.shape[0]) ]

def plot_event_plotly(event, tracks, unrecognized_tracks=None, filename="retina"):
  import plotly.graph_objs as go
  import plotly.tools as tls
  import plotly.plotly as py

  def with_return_to_origin(t):
      t1 = np.zeros(t.shape[0] * 2)
      t1[1::2] = t
      return t1

  hits_3d = go.Scatter3d(
      x = event[:, 2],
      y = event[:, 0],
      z = event[:, 1],
      mode = "markers",
      marker = {"size" : 3.0}
  )

  tracks_3d = go.Scatter3d (
      x = with_return_to_origin(tracks[:, 2]),  # x coords
      y = with_return_to_origin(tracks[:, 0]),  # y coords
      z = with_return_to_origin(tracks[:, 1]),  # z coords
      mode = 'lines',      # (!) draw lines between coords (as in Scatter)
      line = dict(
          color = "green",
          width=2
        )
  )

  tracks_unrecognised_3d = go.Scatter3d (
      x = with_return_to_origin(unrecognized_tracks[:, 2]),  # x coords
      y = with_return_to_origin(unrecognized_tracks[:, 0]),  # y coords
      z = with_return_to_origin(unrecognized_tracks[:, 1]),  # z coords
      mode = 'lines',      # (!) draw lines between coords (as in Scatter)
      line = dict(
          color = "red",
          width=2
        )
  )

  data=[hits_3d, tracks_3d, tracks_unrecognised_3d]

  box = mininal_sq(event)

  layout = go.Layout (
    title="Retina"
    # scene = go.Scene(
    #   xaxis=dict(range=box[0]),
    #   yaxis=dict(range=box[1]),
    #   zaxis=dict(range=box[2])
    # )
  )

  fig = go.Figure(data=data, layout=layout)

  return fig