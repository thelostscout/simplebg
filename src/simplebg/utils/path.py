from os.path import dirname, abspath
import inspect
import simplebg


def get_default_path():
    # get the path of simplebg init file
    init_path = abspath(inspect.getfile(simplebg))
    # get the directory two levels above the init file
    return dirname(dirname(dirname(init_path)))
