import capnp
import inspect
import nlabel.io.arriba
from pathlib import Path


_schema = None


def load_schema():
    global _schema
    if _schema is None:
        p = Path(inspect.getfile(nlabel.io.arriba)).parent
        _schema = capnp.load(str(p / 'schema' / 'archive.capnp'))
    return _schema
