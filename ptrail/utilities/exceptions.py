"""
    This file contains all the custom designed exception headers.
    There is nothing here but the exception headers and pass written inside them.
    The purpose of the file is to store all exceptions in one place.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""


class NoHeadersException(Exception):
    pass


class MissingColumnsException(Exception):
    pass


class DataTypeMismatchException(Exception):
    pass


class MandatoryColumnException(Exception):
    pass


class MissingTrajIDException(Exception):
    pass


class NotAllowedError(Exception):
    pass
