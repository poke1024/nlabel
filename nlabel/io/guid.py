import uuid
import time


def _ts_guid():
    u4 = str(uuid.uuid4()).upper()
    ts = hex(time.time_ns())[2:].upper()
    return f"{u4}-{ts}"


def archive_guid():
    return _ts_guid()


def tagger_guid():
    return _ts_guid()


def text_guid():
    return _ts_guid()
