from _config import *

import numpy as np

import matplotlib.pyplot as plt

def gen_event(particles, pseudo_rapidity, primary_vertex):
  particles_n = particles.rvs()
  pr = pseudo_rapidity.rvs(size = particles_n)

  theta = 2.0 * np.arctan(np.exp(-pr))
  phi = np.random.uniform(0.0, 2 * np.pi, size = particles_n)

  ns = np.ndarray(shape = (particles_n, 3), dtype='float64')

  z = np.cos(theta)
  y = np.sin(theta) * np.cos(phi)
  x = np.sin(theta) * np.sin(phi)

  ns[:, 0] = x
  ns[:, 1] = y
  ns[:, 2] = z

  z0 = primary_vertex.rvs()

  return ns, z0

def intersect(ns, z0, layers, inner_r, outer_r, reaction_prob, min_hits = 2):
  hits = list()
  traceable = list()

  for i in xrange(ns.shape[0]):
    h = np.ndarray(shape=(layers.shape[0], 3), dtype='float64')

    a = (layers - z0) / ns[i, 2]

    h[:, 0] = ns[i, 0] * a
    h[:, 1] = ns[i, 1] * a
    h[:, 2] = layers

    r = h[:, 0] ** 2 + h[:, 1] ** 2

    intersected = (r > inner_r ** 2) & (r < outer_r ** 2) & (h[:, 2] > z0)
    h = h[intersected, :]

    interacted = np.random.binomial(1, reaction_prob, size=h.shape[0]) == 0
    h = h[interacted, :]

    hits.append(h)
    traceable.append(np.count_nonzero(interacted) >= min_hits)

  hits = np.vstack(hits)
  traceable = np.hstack(traceable)

  return hits, ns[traceable, :]

def noise_hits(layers, noise_n, inner_r, outer_r):
  length = np.max(layers)
  velo_volume = length * np.pi * (outer_r * outer_r - inner_r * inner_r)
  block_volume = (4.0 * outer_r ** 2) * length

  compensation = block_volume / velo_volume

  n = int(noise_n.rvs() * compensation)

  hits = np.ndarray(shape=(n, 3), dtype='float64')
  hits[:, 2] = layers[np.random.choice(layers.shape[0], size = n)]
  hits[:, 1] = np.random.uniform(-outer_r, outer_r, size = n)
  hits[:, 0] = np.random.uniform(-outer_r, outer_r, size = n)

  r = hits[:, 0] * hits[:, 0] + hits[:, 1] * hits[:, 1]
  in_velo = (r > inner_r) & (r < outer_r)

  return hits[in_velo, :]

def plot_velo(plt, velo_config, layers, z0, hits, zs, dx, dz):
  velo_l = velo_config.length
  velo_r = velo_config.inner_radius
  velo_R = velo_config.outer_radius

  plt.scatter(zs, hits)

  for i in range(layers.shape[0]):
    zs = layers[i]
    plt.plot([zs, zs], [velo_r, velo_R], "--", color ="green")
    plt.plot([zs, zs], [-velo_r, -velo_R], "--", color ="green")

  plt.plot([0, velo_l], [0, 0], color="red", linewidth=2)
  plt.plot([0, velo_l], [velo_r, velo_r], color="red", linewidth=1)
  plt.plot([0, velo_l], [-velo_r, -velo_r], color="red", linewidth=1)
  plt.plot([0, velo_l], [velo_R, velo_R], color="red", linewidth=1)
  plt.plot([0, velo_l], [-velo_R, -velo_R], color="red", linewidth=1)

  for i in range(dx.shape[0]):
    dxdz = dx[i] / dz[i]
    plt.plot([z0, velo_l], [0, (velo_l - z0) * dxdz], "--", color="blue")

  return plt

def monte_carlo(events_number, config = 'config/mc.json', plot_dir = None):
  if hasattr(config, 'keys'):
    mc = from_config(config)
  else:
    mc = read_config(config)

  velo = mc.velo
  scattering = mc.scattering
  interaction = mc.interaction

  particles = get_distribution(scattering.number_of_particles)
  pseudo_rapidity = get_distribution(scattering.pseudo_rapidity)
  primary_vertex = get_distribution(scattering.primary_vertex)

  hit_noise = get_distribution(interaction.hit_noise)
  detector_noise = get_distribution(interaction.detector_noise)

  layers = np.linspace(0.0, velo.length, num = velo.layers + 1)[1:]

  events = list()
  for event_id in xrange(events_number):
    ns, z0 = gen_event(particles, pseudo_rapidity, primary_vertex)

    detected, ns_ = intersect(ns, z0, layers,
                              velo.inner_radius, velo.outer_radius,
                              interaction.reaction_probability,
                              min_hits = interaction.min_hits_to_trace)

    detected[:, 0:2] += hit_noise.rvs(size=(detected.shape[0], 2))
    noise = noise_hits(layers, detector_noise, velo.inner_radius, velo.outer_radius)
    hits = np.vstack([detected, noise])

    events.append(Event(hits = hits, tracks = ns_, z0 = z0))

    if plot_dir is not None:
      fig, axes = plt.subplots(2, 1, figsize=(14, 18))

      plot_velo(axes[0], velo, layers, z0, hits[:, 0], hits[:, 2], ns_[:, 0], ns_[:, 2])

      axes[0].set_xlim([0, velo.length])
      axes[0].set_ylim([-1.5 * velo.outer_radius, 1.5 * velo.outer_radius])
      axes[0].set_xlabel("Z")
      axes[0].set_ylabel("X")

      plot_velo(axes[1], velo, layers, z0, hits[:, 1], hits[:, 2], ns_[:, 1], ns_[:, 2])

      axes[1].set_xlim([0, velo.length])
      axes[1].set_ylim([-1.5 * velo.outer_radius, 1.5 * velo.outer_radius])
      axes[1].set_xlabel("Z")
      axes[1].set_ylabel("Y")

      plt.suptitle("Event %d" % event_id)

      import os.path as osp
      plt.savefig(osp.join(plot_dir, "velo_event_%d.png" % event_id))

  return events




