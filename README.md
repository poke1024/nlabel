![Logo](doc/nlabel_logo.png)

> nlabel is currently alpha software and in an early stage of development.

nlabel is a library for generating, storing and
retrieving tagging information and embedding vectors
from various nlp libraries through a unified interface.

nlabel is also a system to collate results from various
taggers and keep track of used models and configurations.

Apart from its standard persistence through sqlite and
json files, nlabel's binary arriba format combines
especially low storage requirements with high performance
(see benchmarks below).

Through arriba, nlabel is thus especially suitable for

* inspecting many features on few documents
* inspecting few features on many documents

To support external tool chains, nlabel supports exporting
to [REFI-QDA](https://www.qdasoftware.org/downloads-project-exchange/).

## Quick Start

Processing text works occurs in two steps. First, a
NLP instance is built from an existing NLP pipeline:

```python3
from nlabel import NLP

import spacy

nlp = NLP(spacy.load("en_core_web_sm"), renames={
    'pos': 'upos',
    'tag': 'xpos'
}, require_gpu=False)
```

In the example, above nlp now contains a pipeline based
on spacy's `en_core_web_sm` model. We instruct `nlp` to
generate embedding vectors via `vectors`, and to rename
two tags, namely `pos` to `upos` and `tag` to `xpos`.

In the next step, we run the pipeline and look at its
output:

```python3
doc = nlp(
    "If you're going to San Francisco,"
    "be sure to wear some flowers in your hair.")

for sent in doc.sentences:
    for token in doc.tokens:
        print(token.text, token.upos, token.vector)
```

You can ask a `doc` which tags it carries, by calling `nlabel.tags(doc)`.
In the example above, this would give:

```python3
['dep', 'ent_iob', 'lemma', 'morph', 'sentence', 'token', 'upos', 'xpos']
```

In the following sections, some of internal concepts
will be explained. To get directly to code that will
generate archives for document collections, skip to
[Importing a CSV to a local archive](import-csv).

### Tags and Labels

nlabel handles everything as tags, even it is has no label.
That means that nlabel regards `tokens` and `sentences` as
as tags with labels. Tags can both be iterated but also asked
for labels. Tags can also be regarded as containers that contain
other tags. The following examples illustrate the concepts:

```python3
for ent in doc.ents:
    print(ent.label, ent.text)
```

outputs

```GPE San Francisco```,

while

```python3
for ent in doc.ents:
    for token in ent.tokens:
        print(ent.label, token.text, token.xpos)
```

outputs

```
GPE San PROPN
GPE Francisco PROPN
```

### NLP engines

To plug in a different nlp engine, set `nlp` differently:

```python3
import stanza
nlp = NLP(stanza.Pipeline('en'))
```

Since we renamed `tag` and `pos`, in the spacy example above,
this would work without additional work.

At the moment nlabel has implementations for spacy, stanza,
flair and deeppavlov. You can also write your own nlp data
generators (based on `nlabel.nlp.Tagger`).

While `NLP` usually auto-detects the type of NLP parser you
provide it, there are specialized constructors (`NLP.spacy`,
`NLP.flair`, etc.) that cover some border cases.

### Saving and Loading Documents

Documents can be saved to disk:

```python3
doc.save("path/to/file")
```

By default, this will generate a json-based format that should
be easy to parse, even if you do decide to not use nlabel after
this point - see [bahia json documentation](doc/bahia_json.txt).

Of course, you can also use nlabel to load its own documents:

```python3
from nlabel import Document

with Document.open("path/to/file") as doc:
    for sent in doc.sentences:
        for token in sent.tokens:
            print(token.text, token.upos, token.vector)
```

### Working with Archives

To store data from multiple taggers and texts, the approach
from the last section would generate lots of separate files.
nlabel offers a much better alternative through `Archive`s.

There will be more detailed info on archives later on,
for now, here is a quick run-through of how to use them.

#### A first example

This creates (or opens an existing) archive using the
`carenero` engine (details later on), and adds a newly
parsed document to it.

```
with open_archive("/path/to/archive", engine="carenero") as archive:
    doc = nlp(text)
    archive.add(doc)
```

Opening the archive later would allow us to retrieve all documents:

```
with open_archive("/path/to/archive", "r") as archive:
    for doc in archive.iter():
        print(doc.text)
```

Archives know some more things like the number of documents -
use `len(archive)` - or information about its taggers (see next
section).

#### Multiple Taggers

Things get interesting when using more than one tagger, e.g.:

```
with open_archive("/path/to/archive", engine="carenero") as archive:
    archive.add(nlp1(text))  # e.g. spacy
    archive.add(nlp2(text))  # e.g. stanza
```

In such an archive, calling ```archive.iter()``` will produce
an error:

```
there are 2 taggers with conflicting tag names in this archive,
please use a selector
```

The reason for this error message is that spacy's and stanza's
tag names clash, and nlabel would not know how to deciper `doc.tokens`
to map either to spacy's or stanza's `token` data.

To resolve this issue, we can specify which tagger to use in `iter`.

To do this, we can first ask the archive for the taggers it knows
by calling `archive.taggers`. Each tagger carries a unique signature
that identifies it. For example, `print(archive.taggers[0])` might
the following signature:

```
env:
  machine: arm64
  platform: macOS-12.1-arm64-arm-64bit
  runtime:
    nlabel: 0.0.1.dev0
    python: 3.9.7
library:
  name: spacy
  version: 3.2.1
model:
  lang: en
  name: core_web_sm
  version: 3.2.0
renames:
  pos: upos
  tag: xpos
type: nlp
vectors:
  token:
    type: native
```

To iterate over documents getting tag data from this tagger, we
can use `archive.iter(archive.taggers[0])`.

More commonly, we want to select a tagger based on its attributes,
not on its index in an archive. To do this, we can use a MongoDB
style query syntax:

```
spacy_tagger = archive.taggers[{
    'library': {
        'name': 'spacy'
    }
}]
```

This will return the tagger, that carries the name 'spacy' in the
'library' section of its signature. If there are no or multiple
such taggers, we will get a `KeyError`.

As shorthand for the query above, you can also use:

```
spacy_tagger = archive.taggers[{
    'library.name': 'spacy'
}]
```

### Mixing and Bridging Taggers

What happens if we want not exactly one tagger, but the output from multiple
taggers.

`Archive.iter()` also allows to specify single tags and even rename them.

Using `spacy_tagger` from the last section and a new `stanza_tagger`:

`for doc in archive.iter(
    spacy_tagger.sentence,
    spacy_tagger.xpos,
    stanza_tagger.xpos.to('st_xpos'))):`

With these docs, we now can access spacy's `sentence` and `xpos` tags,
but also stanza's `xpos` tag, which we rename to `st_xpos` to avoid a
name clash with spacy's `xpos' tag:

```
    for token in doc.tokens:  # spacy tokens
        print(token.xpos)  # spacy xpos
        print(token.st_xpos)  # stanza xpos
```

Note that this only works, if stanza's tokenization for a token exactly
matches that of spacy.

### The Design of nlabel and Inherent Quirks

nlabel does not differentiate between tags and structuring entities
such as sentences and tokens. All of them are the same concept to
nlabel: labeled spans, that can be containers to other spans.

What can look like a bug at times, is a very conscious design
decision: nlabel is completely agnostic to tags in terms of knowing
only a single concept that it applies to *everything*.

Due to this design, there are various formulations in the API that
are perfectly valid but rather confusing.

Obviously, it is desirable to write code that avoids these valid but
quirky formulations.

#### Anything is a span with a label

The code below will look for a tag called "pos" that
is perfectly aligned with the current token. If such a tag exists,
nlabel considers it to be the "token's pos tag", and will
return this tag's label.

```
for token in doc.tokens:
    print(token.pos)
```

Here is a quirky twist on the code above:

```
for token in doc.tokens:
    print(token.sentence)
```

This is allowed. The code will do the same thing as above: first it
looks for a tag called "sentence" that is perfectly aligned with the
current token. If such a tag exists, its label is returned.

Since the "sentence" tags provided by nlp libraries carry no labels,
and "sentence" tags are not aligned to "token" tags, this will fail
at step one or two, and therefore just return an empty label. Still,
it is valid in terms of nlabel's concepts.

#### Using the "label" attribute

```
for ent in sentence.ents:
    print(ent.label)
```

The following code does exactly the same thing (avoid using it):

```
for ent in sentence.ents:
    print(ent.ent)
```


### Label Types

There are four label types in nlabel:

|          | description                    | notes                 | type        |
|----------|--------------------------------|-----------------------|-------------|
| `str`    | label values as one string (*) | ignores scores        | str         |
| `strs`   | string list of label values    | ignores scores        | List[str]   |  
| `labels` | all labels                     | label = value + score | List[Label] | 

(*) multiple values are separated by "|"

`strs` and `labels` are suitable for getting output from taggers that return
multiple labels. 

The default type is `str`. The exception to this rule are morphology tags (e.g.
spacy's `morph` and stanza's `feats`, which default to `strs`).

To specify label types, use the `.to(label_type=x)` method on tags, when specifying
them to `Archive.iter` or `Group.view`.

### Groups and Views

`Group`s are an underlying building block of nlabel.
You might not encounter them directly.

A group contains data from multiple taggers for *one* shared text.
If you need to collect data for multiple texts, use archives.

`Document`s can be combined into `Group`s, which will
then contain information from multiple taggers:

```python3
from nlabel import Group

group = Group.join([doc1, doc2])
```

`Group`s have a `view` method that works similar to the `iter`
method available in `Archive`s.

### Computing Embeddings

The following code uses a spacy model to generate token
vectors from spacy's native `vector` attribute:

```python3
nlp = NLP.spacy(
    spacy_model,
    vectors={'token': nlabel.embeddings.native})
```

Spacy's `vector` attribute is usually filled via spacy's own
[`Tok2Vec` and `Transformer` components](https://spacy.io/usage/embeddings-transformers)
or external extensions such as
[spacy-sentence-bert](https://github.com/MartinoMensio/spacy-sentence-bert).

Alternatively, the following code constructs a model that
computes transformer embeddings for tokens via flair:

```python3
nlp = NLP.flair(
    vectors={'token': nlabel.embeddings.huggingface(
        "dbmdz/bert-base-german-cased", layers="-1, -2")},
    from_spacy=spacy_model)
```

`from_spacy` indicates that sentence splitter and tokenizer
should be taken from the provided spacy model.

### Archives

#### Engines

nlabel comes with three different persistence engines:

* `carenero` is for collecting data, esp. in a batch setting - by
supporting restartability and transaction safety, and enabling
export of full data or sub sets of it into bahia or arriba.
* `bahia` is suitable for archival purposes, as it is just
a thin wrapper around a zip of human-readable json files; it is not
the ideal format for exports.
* `arriba` is a binary format optimized for read performance,
it is suitable for data analysis; it is not suitable for exports.

#### Storage Size

The following graph shows data from a real-world dataset,
consisting of 18861 texts (125.3 MB text data), tagged with
4 taggers and a total of 31 tags (no embedding data). Y
axis shows size in GB (note logarithmic scale). REFI-QDA
is roughly 100 times the size of arriba.

![storage size requirements for different engines](doc/storage_size.svg)

#### Random Access Speeds

The exact speed of `arriba` depends on the task and data,
but but often `arriba` performs 10 to 100 times
faster than `bahia` and `carenero` on real-world projects.
From the same data set as earlier (when extracting all POS
tags from one of 4 taggers over 2000 documents):

![access times for different engines](doc/storage_speed.svg)

The `carenero/ALL` benchmarks shows the time when accessing
all tags from all taggers through `carenero`.

#### More Engine Details

These engines support storing both tagging data and
embedding vectors. In the ordering above, they go from slower to faster.

|                       | carenero  | bahia | arriba |
|-----------------------|-----------|-------|--------|
| data collection       | +         | -     | -      |
| exporting             | +         | -     | -      |
| read speeds           | -         | -     | +      |
| suitable for archival | -         | +     | -      |

(*) bahia supports writes, but does not avoid adding duplicates
or support proper restartability in batch settings, i.e. it is
not suited to incremental updates.

### Additional Examples

#### <a name="import-csv">Importing a CSV to a local archive</a>

Create a `carenero` archive from a CSV:

```python3
from nlabel.importers import CSV

import spacy

csv = CSV(
    "/path/to/some.csv",
    keys=['zeitung_id', 'text_type_id', 'filename'],
    text='text')
csv.importer(spacy.load("en_core_web_sm")).to_local_archive()
```

This will create an archive located in the same folder as the
CSV. The code above is restartable, i.e. it is okay to interrupt
and continue later - it will not add duplicate entries.

Once the archive has been created, one can either use it
directly, e.g. iterating its documents:

```python3
from nlabel import open_archive

with open_archive("some/archive.nlabel", mode="r") as archive:
    for doc in archive.iter(some_selector):
        for x in doc.tokens:
            print(x.text, x.xpos, x.vector)
```

Or, one can save the archive to different formats for faster
traversal:

```python3
archive.save("demo2", engine="bahia")
archive.save("demo3", engine="arriba")
```

The `open_archive` call from above works with all archive types.

Note that the `iter` call on archives takes an optional view
description that allows picking/renaming tags as described earlier.

#### Exporting to a remote archive

For larger jobs, it is often useful to separate computation and storage,
and to allow multiple computation processes (both often applies to
GPU cluster environments). Since `carenero`'s sqlite is bad at handling
concurrent writes, the solution is starting a dedicated web service that
handles the writing on a dedicated machine.

On machine A, start an archive server (it will write a carenero archive
to the given path):

```bash
python -m nlabel.importers.server /path/to/archive.nlabel --password your_pwd
```

On machine B, you can start one or multiple importers writing to that remote
archive. Modifying the example from the local archive:

```python3
from nlabel import RemoteArchive

remote_archive = RemoteArchive("http://localhost:8000", ("user", "your_pwd"))
csv.importer(spacy.load("en_core_web_sm")).to_remote_archive(
    remote_archive, batch_size=8)
```

#### Exporting REFI-QDA

The following code exports `ent` tags to a [REFI-QDA](https://www.qdasoftware.org/downloads-project-exchange/) project.

```
from nlabel import NLP

import spacy
nlp = NLP(spacy.load("en_core_web_lg"))
text = 'some longer text...'
doc = nlp(text)

doc.save_to_qda(
    "/path/to/your.qdp", {
        'tagger': {
        },
        'tags': {
            'ent'
        }
    })
```

A `save_to_qda` method is also part of `cantenero` and `bahia` archives.
