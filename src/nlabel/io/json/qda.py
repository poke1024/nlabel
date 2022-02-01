import collections
import contextlib
import datetime
import yaml
import uuid
import json
import shutil
import hashlib
import functools
import orjson
import os
import dbm
import zipfile

from lxml import etree
from pathlib import Path
from tqdm import tqdm

from nlabel import version
from nlabel.io.selector import make_selector
from nlabel.io.bahia.label import factories as label_factories


def isotime():
    dt = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')  # e.g. 2022-01-26T12:36:38Z


def zip_dir(zip_path: Path, base_path: Path):
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:

            for dirname, subdirs, files in os.walk(base_path):
                for filename in files:
                    p = Path(dirname) / filename
                    zf.write(p, arcname=str(p.relative_to(base_path)))
    except:
        zip_path.unlink(missing_ok=True)
        raise


class GuidFactory:
    def __init__(self, db, check_guid=False):
        self._db = db
        self._check_guid = check_guid

    def make(self):
        guid = str(uuid.uuid4()).upper()
        if self._check_guid:
            if guid in self._db:
                raise RuntimeError(
                    f"duplicate GUID {guid}")
            self._db[guid] = "X"
        return guid


class Code:
    def __init__(self, prod, guid, name):
        self._prod = prod
        self._guid = guid
        self._name = name

    @property
    def guid(self):
        return self._guid

    @property
    def exported_name(self):
        spec = self._prod.spec
        return f"{self._name} [{spec['library']['name']}]"

    @property
    def color(self):
        s = json.dumps(self._prod.spec, sort_keys=True)
        key = hashlib.blake2b(s.encode("utf8"), digest_size=3).hexdigest()
        return f"#{key}"

    def write_xml(self, xf):
        code_args = dict(
            name=self.exported_name,
            color=self.color,
            isCodable="true",
            guid=self._guid,)

        with xf.element('Code', **code_args):
            description = etree.Element('Description')
            description.text = yaml.dump(self._prod.spec)
            xf.write(description)


class Tagger:
    def __init__(self, guids, spec):
        self._guids = guids
        self._spec = spec
        self._codes = {}

    @property
    def spec(self):
        return self._spec

    @property
    def codes(self):
        return list(self._codes.values())

    def get_code(self, tag_name):
        code = self._codes.get(tag_name)
        if code is None:
            code = Code(self, self._guids.make(), tag_name)
            self._codes[tag_name] = code
        return code


def make_label_text(label):
    value = label['value']
    score = label.get('score')
    if 'score' in label:
        return f'{value} ({score})'
    else:
        return f'{value}'


class TextCache:
    def __init__(self, db):
        self._db = db
        self._n = 0

    def __len__(self):
        return self._n

    def add(self, external_key, doc_guid, code_guid, tag_data):
        self._db[str(self._n)] = orjson.dumps({
            'external_key': external_key,
            'doc_guid': doc_guid,
            'code_guid': code_guid,
            'tags': tag_data
        })
        self._n += 1

    def __iter__(self):
        for k in self._db.keys():
            data = orjson.loads(self._db[k])
            yield TextSource(**data)


@contextlib.contextmanager
def open_cache(path: Path, name: str, klass):
    cache_path = path.with_suffix(f'.{name}.cache')
    full_cache_path = cache_path.parent / (cache_path.name + ".db")

    for p in (cache_path, full_cache_path):
        if p.exists():
            raise RuntimeError(f"{p} must not exist")

    try:
        with dbm.open(str(cache_path), 'n') as db:
            yield klass(db)
    finally:
        full_cache_path.unlink(missing_ok=True)


class TextSource:
    def __init__(self, external_key, doc_guid, code_guid, tags):
        self._external_key = external_key
        self._doc_guid = doc_guid
        self._code_guid = code_guid
        self._tag_data = tags

    @property
    def xml_name(self):
        name = self._external_key if self._external_key else self._doc_guid
        if not isinstance(name, str):
            name = json.dumps(name, sort_keys=True)
        return name

    def write_xml(self, xf, guids, user_guid):
        now = isotime()

        with xf.element(
                'TextSource',
                creatingUser=user_guid,
                modifyingUser=user_guid,
                name=self.xml_name,
                creationDateTime=now,
                modifiedDateTime=now,
                plainTextPath=f"internal://{self._doc_guid}.txt",
                guid=self._doc_guid):

            children = collections.defaultdict(list)
            for i, tag in enumerate(self._tag_data):
                parent = tag.get('parent')
                if parent is not None:
                    children[parent].append(i)

            for i, tag in enumerate(self._tag_data):
                start = tag.get('start')
                end = tag.get('end')
                if start is None or end is None:
                    continue

                with xf.element(
                        'PlainTextSelection',
                        guid=guids.make(),
                        creatingUser=user_guid,
                        modifyingUser=user_guid,
                        name=f"{start},{end}",
                        startPosition=f"{start}",
                        endPosition=f"{end}",
                        creationDateTime=now,
                        modifiedDateTime=now):

                    # FIXME: also traverse children[i]

                    labels = tag.get('labels')
                    if labels:
                        label_text = ';'.join([make_label_text(x) for x in labels])
                        description = etree.Element('Description')
                        description.text = label_text
                        xf.write(description)

                    with xf.element(
                            'Coding',
                            guid=guids.make(),
                            creatingUser=user_guid,
                            creationDateTime=now):

                        xf.write(etree.Element(
                            'CodeRef', targetGUID=self._code_guid))


class Writer:
    def __init__(self, xf, sel, guids, user_guid, sources_path, text_cache):
        self._xf = xf
        self._sel = sel
        self._guids = guids
        self._user_guid = user_guid
        self._sources_path = sources_path
        self._text_cache = text_cache

        self._taggers = {}

    def _get_tagger(self, spec):
        key = json.dumps(spec, sort_keys=True)
        val = self._taggers.get(key)
        if val is None:
            val = Tagger(self._guids, spec)
            self._taggers[key] = val
        return val

    def _get_code(self, tagger_spec, tag_name):
        return self._get_tagger(tagger_spec).get_code(tag_name)

    @property
    def codes(self):
        for prod in self._taggers.values():
            for code in prod.codes:
                yield code

    def _add_text(self, tagger_index, tag_form, tag_data, doc, taggers, doc_guid):
        code = self._get_code(
            taggers[tagger_index]['tagger'], tag_form.name.external)
        self._text_cache.add(
            doc.external_key, doc_guid, code.guid, tag_data)

    def write_sources(self):
        # FIXME this takes forever on large datasets. investigate.

        for source in tqdm(self._text_cache, total=len(self._text_cache), desc="writing XML"):
            source.write_xml(self._xf, self._guids, self._user_guid)

    def add(self, doc):
        data = doc.data

        doc_guid = self._guids.make()
        with open(self._sources_path / f'{doc_guid}.txt', 'w') as f:
            f.write(doc.text)

        self._sel.build(
            data['taggers'],
            functools.partial(
                self._add_text,
                doc=doc, taggers=data['taggers'], doc_guid=doc_guid))


class Exporter:
    def __init__(self, path: Path, *selectors, exist_ok=False):
        path = Path(path)

        if path.suffix != '.qdpx':
            raise RuntimeError(f"path '{path}' should in .qdpx")

        self._temp_path = path.with_suffix(".qdp")
        self._zip_file_path = path.with_suffix('.qdpx')

        for p in (self._temp_path, self._zip_file_path):
            if p.exists():
                if exist_ok:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                else:
                    raise RuntimeError(f"file {p} already exists.")

        self._temp_path.mkdir()
        self._sources_path = self._temp_path / 'sources'
        self._sources_path.mkdir()

        self._sel = make_selector(label_factories, selectors)

        self._text_sources = []

    @contextlib.contextmanager
    def writer(self, project_name="untitled"):
        attr_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance", "schemaLocation")

        project_qname = {
            attr_qname:
                'urn:QDA-XML:project:1.0 http://schema.qdasoftware.org/versions/Project/v1.0/Project.xsd'
        }

        nsmap = {
            None: 'urn:QDA-XML:project:1.0',
           'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        with open_cache(self._temp_path, 'guids', GuidFactory) as guids:

            user_guid = guids.make()
            now = isotime()

            try:
                with etree.xmlfile(str(self._temp_path / f'{project_name}.qde'), encoding='utf-8') as xf:
                    xf.write_declaration(standalone=True)

                    with xf.element(
                            'Project', project_qname, name=project_name,
                            origin=f'nlabel {version.__version__}',
                            modifiedDateTime=now, nsmap=nsmap):

                        with xf.element('Users'):
                            user = etree.Element('User', name="nlabel", guid=user_guid)
                            xf.write(user)

                        with open_cache(self._temp_path, 'texts', TextCache) as text_cache:
                            writer = Writer(
                                xf, self._sel, guids, user_guid,
                                self._sources_path, text_cache)
                            yield writer

                            with xf.element('CodeBook'):
                                with xf.element('Codes'):
                                    for code in writer.codes:
                                        code.write_xml(xf)

                            with xf.element('Sources'):
                                writer.write_sources()

                zip_dir(
                    self._zip_file_path, self._temp_path)

            finally:
                shutil.rmtree(self._temp_path)
