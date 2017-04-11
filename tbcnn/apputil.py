from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import datetime as dt

from .config import hyper

logger = logging.getLogger(__name__)


class MsecFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s


def initialize(*args, **kwargs):
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setFormatter(MsecFormatter(fmt='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f'))
    # add ch to logger
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    rootLogger.addHandler(ch)

    hyper.initialize(*args, **kwargs)
