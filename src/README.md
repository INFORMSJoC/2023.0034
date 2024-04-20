# Folder Description
The main code of IAAS is in folder `pyIAAS`.
The code in `compare.py`, `generate_report.py` and `search_net.py` is related numerical experiments.

# Installation
First, install the GPU version of [PyTorch](https://pytorch.org/get-started/locally/) (1.13.0+cu116 in our case).

Second, install all the dependencies listed in ```requirements.txt```

## Output file explanations
In the search process of IAAS, the output files are as follows:
- model.db: detailed records of all searched models
- each searched model contains:
  - prediction results of the test dataset
  - transformation table
  - model parameters of type ```.pth```
  - training loss curve 
  
## Customized module list
To extend the code in `pyIAAS` to search for customized neural architectures, Please add the customized modules to the configuration file. 

The modules used in the searching process is given in the configuration
file. The default configuration is 
```json
{
  "MaxLayers": 50,
  "timeLength": 168,
  "predictLength": 24,
  "IterationEachTime": 50,
  "MonitorIterations": 40,
  "NetPoolSize": 5,
  "BATCH_SIZE": 256,
  "EPISODE": 200,
  "GPU": true,
  "OUT_DIR": "out_dir",
  "Modules" : {
  "dense": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  },
  "rnn":{
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  },
  "lstm":{
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  },
  "conv": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  }
  }
}
```

Note: The LSTM module is super compares to the original implementation of pytorch, since the pytorch code is optimized for cuda parallelization. The running time of LSTM in CPU is similar.

The meaning of each term:
- MaxLayers : number of the maximum layers of the searched neural architecture
- timeLength : length of the input time-series data
- predictLength : prediction time length, e.g., two-hour ahead
- IterationEachTime : number of the training epochs at each searching episode
- MonitorIterations : epoch interval to print out the training information, e.g., training loss 
- NetPoolSize : size of the net pool
- BATCH_SIZE : batch size used in the training process
- EPISODE : searching times of the reinforcement learning actors
- Pruning : enable pruning functionality during search (Pruning implementation inspired by [Movement Pruning](https://github.com/huggingface/nn_pruning))
- PruningRatio : pruning ratio(topV strategy used here)
- GPU : use GPU or not; if true, the environment should use the GPU version of PyTorch
- OUT_DIR : output directory
- Modules : module information 
  - out_range : list of the output unit number 
  - editable : whether this module can be widened or not

## Extending new module
To create a new module, users should create a subclass of ```pyIAAS.model.module.NasModule```, and implement 
these reserved abstract functions

```python
from pyIAAS.model.module import NasModule
# this is a sample subclassing of NasModule  to
# illustrate how to customize a new module in the pyIAAS package
class NewModule(NasModule):
    @property
    def is_max_level(self):
        # return: True if this module reaches the max width level, False otherwise
        raise NotImplementedError()

    @property
    def next_level(self):
        # return: width of next level
        raise NotImplementedError()

    def init_param(self, input_shape):
        # initialize the parameters of this module
        self.on_param_end(input_shape)
        raise NotImplementedError()

    def identity_module(self, cfg, name, input_shape: tuple):
        # generate an identity mapping module
        raise NotImplementedError()

    def get_module_instance(self):
        # generate a model instance once and use it for the rest procedures
        raise NotImplementedError()

    @property
    def token(self):
        # return: string type token of this module
        raise NotImplementedError()

    def perform_wider_transformation_current(self):
        # generate a new wider module by the wider function-preserving transformation
        # this function is called by layer i and returns the realized random mapping to the IAAS framework for the next layer's wider transformation.
        raise NotImplementedError()

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        # generate a new wider module by the wider function-preserving transformation
        # this function is called by the layer i + 1
        raise NotImplementedError()
```

Add the module information to the configuration file as follows
```json
{
  "MaxLayers": 50,
  "timeLength": 168,
  "predictLength": 24,
  "IterationEachTime": 50,
  "MonitorIterations": 40,
  "NetPoolSize": 5,
  "BATCH_SIZE": 256,
  "EPISODE": 200,
  "GPU": true,
  "OUT_DIR": "out_dir",
  "Modules" : {
  "dense": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
    },
  "new_module": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
    }
  }
}
```
Register this new module in the running code

```python
from pyIAAS import *
from new_module import NewModule
cfg = Config('NASConfig.json')
# register a new module to the global configuration
cfg.register_module('new_module', NewModule)
```