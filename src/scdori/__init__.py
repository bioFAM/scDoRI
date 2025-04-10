import logging

from . import pp
from ._core import (
    config,
    initialize_scdori_parameters,
    load_best_model,
    load_scdori_inputs,
    save_model_weights,
    scDoRI,
    set_seed,
    train_model_grn,
    train_scdori_phases,
)
from ._version import __version__, __version_tuple__

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not logging.root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False
