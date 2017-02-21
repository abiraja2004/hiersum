import os
from functools import partial
from typing import Callable


class LazyFileIO:
    """
    Lazy file IO

    Must be instantiated using a context manager (eq. "with")
    so that files can be closed properly and reliably

    """

    def __init__(self, filename, buf=-1, enc='utf-8'):
        """ """
        if not os.path.isfile(filename):
            raise IOError('Could not find', filename)
        self.context_managed = False
        self._file = filename
        self._fhandle = None
        self.lines = None
        self.buf = buf
        self.enc = enc

    def __enter__(self):
        """ On call with context manager,
        open file and set a line iterator """
        self._fhandle = open(self._file, 'r', self.buf, self.enc)
        self.lines = (line for line in self._fhandle)
        self.context_managed = True
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        """ On exit, close file """
        if exec_type is not None:
            print(exec_type, exec_value, traceback)
        if self._fhandle:
            self._fhandle.close()

    def apply_func(self, func: Callable, *args):
        """ map function over data """
        if not self.context_managed:
            raise Exception('Object must be initialized with context manager')
        try:
            afunc = partial(func, args) if args else func
            yield from map(afunc, self.lines)
        except TypeError as error:
            print(error)

    def apply_funcs(self, *funcs):
        raise NotImplementedError

    def __del__(self):
        """ Unreliable due to Python's garbage collection weirdness """
        if self._fhandle:
            self._fhandle.close()
