import os
from openpilot.system.hardware import TICI
from openpilot.selfdrive.modeld.runners.runmodel_pyx import RunModel, Runtime
assert Runtime

USE_SNPE = int(os.getenv('USE_SNPE', str(int(TICI))))

class ModelRunner(RunModel):
  TINYGRAD = 'TINYGRAD'
  SNPE = 'SNPE'

  def __new__(cls, paths, *args, **kwargs):
    if ModelRunner.TINYGRAD in paths:
      from openpilot.selfdrive.modeld.runners.tinygradmodel import TinygradModel as Runner
      runner_type = ModelRunner.TINYGRAD
    elif ModelRunner.SNPE in paths and USE_SNPE:
      from openpilot.selfdrive.modeld.runners.snpemodel_pyx import SNPEModel as Runner
      runner_type = ModelRunner.SNPE
    else:
      raise Exception("Couldn't select a model runner, make sure to pass at least one valid model path")

    return Runner(str(paths[runner_type]), *args, **kwargs)
