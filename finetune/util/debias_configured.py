"""Provides the `Configured` class.

The `Configured` class provides human readable serialization and version
numbering to subclasses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from collections import OrderedDict
from inspect import signature, Parameter
from warnings import warn

import numpy as np


class Configuration(object):

  def __init__(self, name, version, params):
    self.name = name
    self.version = version
    self.params = params


class Configured(object):
  """A class with a version number or with fields that are either `Configured`
  objects or 'simple' aka json-serializable types.

  Includes some methods for getting a json representation and for pickling.

  Subclasses should have constructor args with matching fields, it may
  have additional fields as long those fields only act as caches, subclasses
  are expected to be effectively stateless and treated as being immutable.

  These classes cam be used as a convenient way to store the specifications
  for models/datasets/metrics/optimizers in way that can be
  serialized to a human readable representation
  """

  @classmethod
  def _get_param_names(cls):
    """Returns all parameter names of the `__init__` method."""
    init = cls.__init__
    if init is object.__init__:
      return []  # No init args

    init_signature = signature(init)
    for param in init_signature.parameters.values():
      if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
        raise ValueError(cls.__name__ + " has kwargs or args in __init__")
    return [p for p in init_signature.parameters.keys() if p != "self"]

  @property
  def name(self):
    return self.__class__.__name__

  @property
  def version(self):
    return 0

  def get_config(self):
    return Configuration(self.name, self.version, self.get_params())

  def get_params(self):
    out = OrderedDict()
    for key in self._get_param_names():
      v = getattr(self, key)
      out[key] = _get_configuration(v)
    return out

  def to_json(self, indent=None):
    return config_to_json(self, indent)

  def __getstate__(self):
    state = {}
    for key in self._get_param_names():
      state[key] = getattr(self, key)
    state["version"] = self.version
    return state

  def __setstate__(self, state):
    if "version" not in state:
      raise RuntimeError(
          "Version should be in state (%s)" % self.__class__.__name__)
    if state["version"] != self.version:
      warn(("%s loaded with version %s, but class version is %s") %
           (self.__class__.__name__, state["version"], self.version))
    del state["version"]
    self.__init__(**state)


def _get_configuration(obj):
  """Transform `obj` into a `Configuration` object or json-serialable type."""

  if isinstance(obj, Configured):
    return obj.get_config()

  obj_type = type(obj)

  if obj_type in (list, set, frozenset, tuple):
    return obj_type([_get_configuration(e) for e in obj])
  elif obj_type in (OrderedDict, dict):
    output = obj_type()
    for k, v in obj.items():
      if isinstance(k, Configured):
        raise ValueError()
      output[k] = _get_configuration(v)
    return output
  elif obj_type in {str, int, float, bool, type(None)}:
    return obj
  if isinstance(obj, np.integer):
    return int(obj)
  if isinstance(obj, np.floating):
    return float(obj)
  else:
    raise ValueError("Can't configure obj " + str(obj_type))


class _ConfiguredJSONEncoder(json.JSONEncoder):
  """`JSONEncoder` that handles `configured` and `configuration` objects"""

  def default(self, obj):
    if isinstance(obj, Configuration):
      if "version" in obj.params or "name" in obj.params:
        raise ValueError()
      out = OrderedDict()
      out["name"] = obj.name
      if obj.version != 0:
        out["version"] = obj.version
      out.update(obj.params)
      return out
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.bool_):
      return bool(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, Configured):
      return obj.get_config()
    elif isinstance(obj, set) or isinstance(obj, frozenset):
      return sorted(obj)  # Ensure deterministic order
    else:
      return super(_ConfiguredJSONEncoder, self).default(obj)


def config_to_json(data, indent=None):
  # sort_keys=False since the configuration objects will dump their
  # parameters in an ordered manner
  return json.dumps(
      data, sort_keys=False, cls=_ConfiguredJSONEncoder, indent=indent)
