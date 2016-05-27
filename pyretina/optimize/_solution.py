from collections import namedtuple

Solution = namedtuple(
  "Solution", [
    'maxima',
    'scores',
    'traces',
    'nfev',
    'njev',
    'nhev'
  ]
)