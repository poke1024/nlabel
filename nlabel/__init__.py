try:
	import capnp
	capnp.remove_import_hook()
except ImportError:
	pass

import nlabel.embeddings

from nlabel.nlp import NLP, Text
from nlabel.io.common import to_path, open_archive as open, RemoteArchive
from nlabel.io.slice import Slice
from nlabel.io.json.property import tags, meta, external_key
from nlabel.io.json.loader import Document
from nlabel.io.json.group import Group
