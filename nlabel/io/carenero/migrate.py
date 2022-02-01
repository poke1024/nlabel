import json

from nlabel.io.carenero.schema import create_session_factory, \
    Text, Tagger, Vector, Vectors, ResultStatus, Result
from tqdm import tqdm
from nlabel import Slice


# ALTER TABLE text ADD COLUMN external_key_type VARCHAR NOT NULL DEFAULT 'json';
# ALTER TABLE nlp RENAME TO tagger;
# ALTER TABLE result RENAME COLUMN nlp_id TO tagger_id;
# ALTER TABLE text ADD COLUMN text_hash_code VARCHAR NOT NULL DEFAULT '';


def migrate_nlp_to_taggers(result):
    assert result.status == ResultStatus.succeeded
    data = json.loads(result.content)
    nlps = data.get('nlps')
    if nlps is not None:
        data = data.copy()
        assert len(nlps) == 1
        del data['nlps']
        assert json.dumps(nlps[0]['nlp'], sort_keys=True) == result.tagger.description
        data['tags'] = nlps[0]['tags']
        result.content = json.dumps(data)
        return True
    else:
        return False


def migrate(path, parallel=None):
    parallel_filter = Slice(parallel)
    session_factory = create_session_factory(path)

    session = session_factory()
    try:
        n_results = session.query(Result).filter(
            Result.status == ResultStatus.succeeded).count()

        results = session.query(Result).filter(
            Result.status == ResultStatus.succeeded).order_by(Result.id).all()

        for i, result in enumerate(tqdm(results, total=n_results)):
            if not parallel_filter(i):
                continue
            if migrate_nlp_to_taggers(result):
                session.commit()
    finally:
        session.close()


def migrate_external_keys(path):
    # convert ["2436020X_1872-01-04_0_5_010", "tagesbericht", "bbz"]
    # to {"filename": "2436020X_1921-04-01_66_150_003", "text_type_id": "tagesbericht", "zeitung_id": "bbz"}

    session_factory = create_session_factory(path)

    session = session_factory()
    try:
        n_texts = session.query(Text).count()

        texts = session.query(Text).order_by(Text.id).all()

        for i, text in enumerate(tqdm(texts, total=n_texts)):
            k_text = text.external_key
            if k_text.startswith('['):
                new_external_key = json.dumps(dict(zip(
                    ['filename', 'text_type_id', 'zeitung_id'], json.loads(k_text))))
                text.external_key = new_external_key
                session.commit()
    finally:
        session.close()
