import os.path as osp
import importlib.util


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location("", pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module