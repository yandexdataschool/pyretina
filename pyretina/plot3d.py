from mayavi import mlab
import numpy as np

def plot_event(event, tracks = None):
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
