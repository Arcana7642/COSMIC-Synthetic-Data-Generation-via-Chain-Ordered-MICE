import torch

try:
    from icecream import install
except ImportError:
    def install():
        return None

torch.set_num_threads(1)
install()

from . import env  # noqa
from .data import *  # noqa
from .env import *  # noqa
from .util import *  # noqa
