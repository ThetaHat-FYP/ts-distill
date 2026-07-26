"""
Logging
=======
A library must not write to a user's stdout uninvited, so every progress
message in ts_distill goes through the standard `logging` module instead of
`print`. By default the `ts_distill` logger has a NullHandler attached, which
means importing the package produces no output at all.

Scripts that WANT to see progress opt in with one call:

    from ts_distill import configure_logging
    configure_logging()               # INFO-level progress to stderr
    configure_logging('DEBUG')        # more detail
    configure_logging('WARNING')      # quiet: only problems

Anything the user explicitly asked to see — `CSVDataLoader.view_data()`, the
`verbose=True` branch of `predict_r_star`, CLI summary tables — still uses
`print`, because there the output IS the return value.
"""

import logging

_ROOT_LOGGER_NAME = 'ts_distill'

# Attach a NullHandler once, so `import ts_distill` never emits anything and
# never triggers the "No handlers could be found" warning.
logging.getLogger(_ROOT_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """
    Return the module-level logger for a ts_distill module.

    Args:
        name (str): Normally `__name__` of the calling module.

    Returns:
        logging.Logger: A logger under the `ts_distill` hierarchy, so a single
        `configure_logging()` call controls output for the whole library.
    """
    return logging.getLogger(name)


def configure_logging(level='INFO', fmt: str = '%(message)s') -> logging.Logger:
    """
    Turn on console output for ts_distill's progress messages.

    Intended for scripts and notebooks. Applications that already configure
    logging themselves should skip this and just set the level on the
    `ts_distill` logger.

    Args:
        level (str | int): Threshold, e.g. 'INFO', 'DEBUG', 'WARNING'.
        fmt   (str):       `logging` format string. The default prints the bare
                           message, matching the framework's previous `print`
                           output.

    Returns:
        logging.Logger: The configured `ts_distill` logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)

    # Replace any handler this function added before, so repeated calls (common
    # in notebooks) do not duplicate every line.
    for handler in [h for h in logger.handlers if getattr(h, '_ts_distill', False)]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    handler._ts_distill = True
    logger.addHandler(handler)

    # Messages are emitted by our own handler; don't let the root logger print
    # them a second time.
    logger.propagate = False
    return logger
