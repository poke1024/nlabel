import contextlib
import concurrent.futures
import requests
import traceback
import orjson
import logging
import queue as queues

from typing import List, Iterator, Union
from typing import NamedTuple

from nlabel.io.carenero.common import TaggerFactory, ExternalKey,\
    ResultFactory, gen_message
from nlabel.io.carenero.schema import Text, Result, ResultStatus
from nlabel.nlp.nlp import NLP as CoreNLP, Text as CoreText
from nlabel.io.common import RemoteArchive
from nlabel.io.json.group import Group


class RemoteResultFactory(ResultFactory):
    def __init__(self, nlp):
        self._tagger_signature = orjson.dumps(
            nlp.signature, option=orjson.OPT_SORT_KEYS)

    def _check_signature(self, signature):
        assert self._tagger_signature == orjson.dumps(
            signature, option=orjson.OPT_SORT_KEYS)

    def _make_succeeded(self, json_data, vectors_data):
        result = {
            'status': ResultStatus.succeeded.name,
            'data': json_data
        }

        if vectors_data:
            dtype = '<f4'
            vdata = {}
            for nlp_vectors_data in vectors_data:
                for k, v in nlp_vectors_data.items():
                    vdata[k] = [x.astype(dtype).tobytes().hex() for x in v]
            result['vectors'] = {
                'data': vdata,
                'dtype': dtype
            }

        return result

    def make_failed(self, err):
        return {
            'status': ResultStatus.failed.name,
            'data': err
        }


class Chunker:
    def __init__(self, queue, nlp, batch_size, chunk_size=None):
        self._queue = queue
        self._nlp = nlp
        self._batch_size = batch_size
        self._chunk_size = chunk_size if chunk_size else batch_size
        self._messages = []

    def flush(self):
        if not self._messages:
            return

        for message, doc in zip(
                self._messages,
                self._nlp.pipe((x.text for x in self._messages), batch_size=self._batch_size)):
            self._queue.put(TagsMessage(
                text_id=message.text_id, text=message.text, doc=doc.collection, err=None))

        self._messages = []

    def push(self, item):
        self._messages.append(item)
        if len(self._messages) >= self._chunk_size:
            self.flush()


class TextMessage(NamedTuple):
    text_id: int
    text: CoreText


class TextProducer:
    def __init__(self, api_url, auth, tagger_id, queue_size, timeout):
        self._api_url = api_url
        self._auth = auth
        self._tagger_id = tagger_id
        self._queue = queues.Queue(maxsize=queue_size)
        self._timeout = timeout

    @property
    def queue(self):
        return self._queue

    def __call__(self, items: List[CoreText]):
        try:
            for text_item in items:

                response = requests.post(f"{self._api_url}/texts", json={
                    'external_key': text_item.external_key,
                    'text': text_item.text,
                    'meta': text_item.meta
                }, auth=self._auth, timeout=self._timeout)
                if response.status_code != 200:
                    logging.info(
                        "could not POST text '{text_item.external_key}'"
                        "({response.status_code}): {response.text}")
                    continue
                text_id = int(response.json()["id"])

                response = requests.get(
                    f"{self._api_url}/taggers/{self._tagger_id}/texts/{text_id}/results", params={
                        'fields': 'id'
                    }, auth=self._auth, timeout=self._timeout)
                if response.status_code == 404:
                    # result does not exist yet, so go ahead and compute it.
                    self._queue.put(TextMessage(text_id, text_item))
                elif response.status_code != 200:
                    logging.info(
                        "unable to retrieve result of text '{text_item.external_key}'"
                        "({response.status_code}): {response.text}")
                    continue

        except:
            traceback.print_exc()
            raise

        finally:
            self._queue.put("STOP")


class TagsMessage(NamedTuple):
    text_id: int
    text: CoreText
    doc: Union[Group, None]
    err: Union[str, None]


class TagsProducer:
    def __init__(self, nlp, queue_size, batch_size):
        self._nlp = nlp
        self._queue = queues.Queue(maxsize=queue_size)

    @property
    def queue(self):
        return self._queue

    def __call__(self, texts_queue: queues.Queue):
        try:
            while True:
                message = texts_queue.get()

                if message == "STOP":
                    break

                message = TagsMessage(text_id=message.text_id, **gen_message(
                    self._nlp, message.text))

                self._queue.put(message)

        except:
            traceback.print_exc()
            raise

        finally:
            self._queue.put("STOP")


class ChunkingTagsProducer:
    def __init__(self, nlp, queue_size, batch_size):
        self._nlp = nlp
        self._queue = queues.Queue(maxsize=queue_size)
        self._batch_size = batch_size

    @property
    def queue(self):
        return self._queue

    def __call__(self, texts_queue: queues.Queue):
        try:
            chunker = Chunker(
                self._queue, self._nlp, self._batch_size)

            while True:
                message = texts_queue.get()

                if message == "STOP":
                    break

                chunker.push(message)

            chunker.flush()

        except:
            traceback.print_exc()
            raise
        finally:
            self._queue.put("STOP")


class Writer:
    def __init__(self, api_url, auth, tagger_id, nlp, timeout):
        self._api_url = api_url
        self._auth = auth
        self._tagger_id = tagger_id
        self._nlp = nlp
        self._timeout = timeout

    def __call__(self, tags_queue: queues.Queue):
        try:
            f = RemoteResultFactory(self._nlp)

            while True:
                message = tags_queue.get()

                if message == "STOP":
                    break

                try:
                    if message.doc is not None:
                        result = f.make_succeeded(message.doc)
                    else:
                        result = f.make_failed(message.err)

                    response = requests.post(
                        f"{self._api_url}/taggers/{self._tagger_id}/texts/{message.text_id}/results",
                        json=result, auth=self._auth, timeout=self._timeout)

                    if response.status_code != 200:
                        logging.info(f"failed to POST result {response.status_code}: {response.text}")

                except:
                    traceback.print_exc()

        except:
            traceback.print_exc()


def batch_add(
        archive: RemoteArchive,
        nlp: CoreNLP, items: Iterator[CoreText],
        batch_size: int = 16,
        timeout: int = None):

    # * prevents actually computing the nlp when doc is already in archive
    # * records errors during nlp processing into archive

    api_url = archive.api_url
    auth = archive.auth

    queue_size = batch_size * 8

    # avoid race condition when creating new tagger.
    response = requests.post(
        f"{api_url}/taggers", json=nlp.signature,
        auth=auth, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"unable to retrieve tagger "
            f"({response.status_code}): {response.text}")
    tagger_id = int(response.json()["id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        text_producer = TextProducer(
            api_url, auth, tagger_id, queue_size, timeout=timeout)
        executor.submit(text_producer, items)

        tp_class = TagsProducer if batch_size <= 1 else ChunkingTagsProducer
        tags_producer = tp_class(nlp, queue_size, batch_size=batch_size)
        executor.submit(tags_producer, text_producer.queue)

        executor.submit(Writer(
            api_url, auth, tagger_id, nlp, timeout=timeout), tags_producer.queue)
