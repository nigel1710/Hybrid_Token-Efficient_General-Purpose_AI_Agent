import logging
import sys


def setup_logging(level: int = logging.INFO):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt)
    # Route ERROR+ to stderr as well
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(err_handler)
