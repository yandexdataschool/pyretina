from collections import namedtuple

Event = namedtuple(
  "Event", [
    'hits',
    'tracks'
  ]
)

VELO = namedtuple(
  "VELO", [
    'length',
    'layers',
    'inner_radius',
    'outer_radius'
  ]
)

Scattering = namedtuple(
  "Scattering", [
    'number_of_particles',
    'pseudo_rapidity',
    'primary_vertex'
  ]
)

Interaction = namedtuple(
  "Interaction", [
    'min_hits_to_trace',
    'reaction_probability',
    'hit_noise',
    'detector_noise'
  ]
)

MC = namedtuple(
  'MC', [
    'velo',
    'scattering',
    'interaction'
  ]
)

def read_config(path = "config/mc.json"):
  import json
  with open(path, 'r') as f:
    config = json.load(f)

  velo_config = config['velo_geometry']

  velo = VELO(
    layers = velo_config['layers'],
    length = velo_config['length'],
    inner_radius = velo_config['inner_radius'],
    outer_radius = velo_config['outer_radius']
  )

  scattering_config = config['scattering']

  scattering = Scattering(
    number_of_particles = scattering_config['number_of_particles'],
    pseudo_rapidity = scattering_config['pseudo_rapidity'],
    primary_vertex = scattering_config['primary_vertex']
  )

  interaction_config = config['interaction']

  interaction = Interaction(
    min_hits_to_trace = interaction_config['min_hits_to_trace'],
    reaction_probability = interaction_config['reaction_probability'],
    hit_noise = interaction_config['hit_noise'],
    detector_noise = interaction_config['detector_noise']
  )

  return MC(
    velo = velo,
    scattering = scattering,
    interaction = interaction
  )

def get_distribution(distribution_params):
  from scipy import stats
  family = distribution_params.get('type', 'norm')

  params = distribution_params.copy()
  del params['type']

  distr = getattr(stats, family)(**params)
  return distr