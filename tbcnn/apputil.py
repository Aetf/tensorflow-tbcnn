from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .config import hyper

logger = logging.getLogger(__name__)


def initialize(*args, **kwargs):
    logging.basicConfig(format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    hyper.initialize(*args, **kwargs)
