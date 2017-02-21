import os
from functools import partial
from typing import Callable, Any


class LazyFileIO:
    """
    Lazy file IO

    Must be instantiated using a context manager (eq. "with")
    so that files can be closed properly and reliably

    Usage:
        apply_func with a function that takes a list of strings
        as its first argument, and then any number of other arguments
        apply_func will call fst_arg_last because it uses map

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

    @staticmethod
    def fst_arg_last(func):
        """ No need to explicitly call this method
            Automatically called when using apply_func
        """
        def inner(*args):
            args = args[1:] + (args[0],)  # unpack then pack tuple
            return func(*args)
        return inner

    def safe_apply_func(self, func, *args):
        if not self.context_managed:
            raise Exception('Object must be initialized with context manager')
        try:
            if not args:
                yield from (func(i) for i in self.lines)
            else:
                yield from(func(i, *args) for i in self.lines)
        except TypeError as error:
            print(error)

    def apply_func(self, func: Callable[[str], Any], *args):  
                    # actually Callable[[str,...], Any]
        """ map function over data """
        if not self.context_managed:
            raise Exception('Object must be initialized with context manager')
        try:
            afunc = func if not args else partial(
                    (LazyFileIO.fst_arg_last(func)), *args)
            yield from map(afunc, self.lines)
        except TypeError as error:
            print(error)

    def apply_funcs(self, *funcs):
        raise NotImplementedError
