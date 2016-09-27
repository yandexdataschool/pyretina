import unittest

import numpy as np

from retina import ReferencePlaneRetinaModel

class ReferencePlaneModelTest(unittest.TestCase):
  def test_compile(self):
    model = ReferencePlaneRetinaModel()

  def test_response(self):
    model = ReferencePlaneRetinaModel()
    model.set_event(np.array([
      [1.0, 1.0, 5.0],
      [0.0, 2.5, 6.0],
      [2.0, 3.0, 7.0],
      [9.0, 5.0, 8.0]
    ]))

    print model.response(0.0, 1.0, 1.0, 0.5)

if __name__ == '__main__':
  unittest.main()
