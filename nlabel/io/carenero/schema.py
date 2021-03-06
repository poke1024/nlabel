import enum
import orjson
import logging

import sqlalchemy
import sqlalchemy.orm

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, BLOB, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship, deferred
from sqlalchemy.sql import func

from pathlib import Path
from cached_property import cached_property


Base = declarative_base()


class ExternalKeyType(enum.Enum):
    str = 0
    json = 1


class Text(Base):
    __tablename__ = 'text'

    id = Column(Integer, primary_key=True)
    guid = Column(String, unique=True, nullable=False)
    external_key = Column(String, unique=True, nullable=False)
    external_key_type = Column(String, nullable=False)
    text = deferred(Column(String, nullable=False))
    text_hash_code = Column(String, nullable=False, index=True)
    meta = deferred(Column(String))
    results = relationship("Result", lazy="dynamic")

    @property
    def decoded_external_key(self):
        if self.external_key_type == 'str':
            return self.external_key
        else:
            return orjson.loads(self.external_key)


class Tagger(Base):
    __tablename__ = 'tagger'

    id = Column(Integer, primary_key=True)
    guid = Column(String, unique=True, nullable=False)
    signature = deferred(Column(String, unique=True))
    results = relationship("Result", lazy="dynamic")
    tags = relationship("Tag", lazy="dynamic")

    @cached_property
    def signature_as_dict(self):
        return orjson.loads(self.signature)


class Tag(Base):
    __tablename__ = 'tag'

    id = Column(Integer, primary_key=True)
    tagger_id = Column(Integer, ForeignKey('tagger.id'))
    tagger = relationship("Tagger", back_populates="tags", uselist=False)
    name = Column(String, nullable=False)
    instances = relationship("TagInstances", lazy="dynamic", back_populates="tag")

    uniq_tagger_name = UniqueConstraint('tagger_id', 'name')


class Vector(Base):
    __tablename__ = 'vector'

    id = Column(Integer, primary_key=True)
    vectors_id = Column(Integer, ForeignKey('vectors.id'), index=True)
    vectors = relationship("Vectors", back_populates="vectors")
    index = Column(Integer)
    data = Column(BLOB)


class Vectors(Base):
    __tablename__ = 'vectors'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    result_id = Column(Integer, ForeignKey('result.id'), index=True)
    result = relationship("Result", back_populates="vectors", uselist=False)
    dtype = Column(String)
    vectors = relationship("Vector", lazy="joined", back_populates="vectors", order_by="Vector.index")


class ResultStatus(enum.Enum):
    succeeded = 0
    failed = 1


class TagInstances(Base):
    __tablename__ = 'tag_instances'

    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.id'))
    result = relationship("Result", back_populates="tag_instances")
    tag_id = Column(Integer, ForeignKey('tag.id'))
    tag = relationship("Tag", uselist=False)
    data = Column(String)

    uniq_result_tag = UniqueConstraint('result_id', 'tag_id')


class Result(Base):
    __tablename__ = 'result'

    id = Column(Integer, primary_key=True)
    text_id = Column(Integer, ForeignKey('text.id'), index=True)
    text = relationship("Text", back_populates="results")
    tagger_id = Column(Integer, ForeignKey('tagger.id'))
    tagger = relationship("Tagger", back_populates="results")
    status = Column(Enum(ResultStatus), index=True)
    data = deferred(Column(String))
    vectors = relationship("Vectors", back_populates="result", lazy="dynamic")
    tag_instances = relationship("TagInstances", back_populates="result", lazy="dynamic")
    time_created = Column(DateTime(timezone=True), server_default=func.now())

    uniq_text_tagger = UniqueConstraint('text_id', 'tagger_id')


def create_session_factory(path, echo=False):
    path = Path(path)

    db_path =path / "database.sqlite"
    logging.info(f"opening {db_path}")

    engine = sqlalchemy.create_engine(
        f'sqlite:///{db_path}', echo=echo)

    # see https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#pysqlite-serializable

    @sqlalchemy.event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before any DDL.
        dbapi_connection.isolation_level = None

    @sqlalchemy.event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN
        conn.exec_driver_sql("BEGIN")

    Session = sqlalchemy.orm.sessionmaker()
    Session.configure(bind=engine)

    Base.metadata.create_all(engine)

    return Session
