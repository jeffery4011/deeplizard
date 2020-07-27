from collections import OrderedDict
from collections import namedtuple
from itertools import product

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
   ,shuffle = [True, False]
   ,device =["cuda","cpu"]
)
Run = namedtuple('Run', params.keys())
print(Run)
runs = []
for v in product(*params.values()):
    runs.append(Run(*v))
print(runs)
runs = RunBuilder.get_runs(params)
# for run in runs:
#     print(run.lr,run.batch_size,run.shuffle,run.device)