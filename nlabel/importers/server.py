from wsgiref.simple_server import make_server

from nlabel.io.carenero.schema import create_session_factory, \
    Text, ResultStatus, Result, Tagger, Vector, Vectors
from nlabel.io.carenero.common import ExternalKey
from nlabel.io.common import ArchiveInfo, text_hash_code
from nlabel.io.carenero.common import json_to_result
from nlabel.io.guid import text_guid, tagger_guid

from sqlalchemy.orm import load_only, lazyload
from falcon_auth2 import AuthMiddleware
from falcon_auth2.backends import BasicAuthBackend

import falcon
import click
import json
import functools
import nlabel.version


def user_loader(attributes, user, password, config):
    if user == config['user'] and password == config['password']:
        return True
    else:
        return False


class PingResource:
    def on_get(self, req, resp):
        resp.text = json.dumps({
            'version': nlabel.version.__version__
        })
        resp.status = falcon.HTTP_200


class TaggersByIdResource:
    def __init__(self, new_session):
        self._new_session = new_session

    def on_get(self, req, resp, tagger_id):
        session = self._new_session()
        try:
            tagger = session.query(Tagger).filter(
                Tagger.id == tagger_id).first()
            if tagger is None:
                resp.status = falcon.HTTP_204
            else:
                resp.status = falcon.HTTP_200
                resp.text = json.dumps(tagger.signature)
        finally:
            session.close()

        resp.status = falcon.HTTP_200


class TaggersResource:
    def __init__(self, new_session):
        self._new_session = new_session

    def on_post(self, req, resp):
        tagger_data = req.media

        session = self._new_session()
        try:
            tagger_json = json.dumps(
                tagger_data, sort_keys=True)

            tagger = session.query(Tagger).filter_by(
                signature=tagger_json).first()
            if tagger is None:
                tagger = Tagger(
                    guid=tagger_guid(),
                    signature=tagger_json)
                session.add(tagger)
                session.commit()
                session.refresh(tagger)

            resp.status = falcon.HTTP_200
            resp.text = json.dumps({
                'id': tagger.id
            })
        finally:
            session.close()


class TextsResource:
    def __init__(self, new_session):
        self._new_session = new_session

    def on_post(self, req, resp):
        text_data = req.media

        invalid_keys = set(text_data.keys()) - {
            'external_key', 'text', 'meta'}
        if invalid_keys:
            raise falcon.HTTPInvalidParam(
                "media", str(invalid_keys))

        external_key = ExternalKey.from_value(
            text_data.get('external_key'))
        text_key = text_data.get('text')

        meta_key = text_data.get('meta')
        if meta_key is None:
            meta_key = ''
        else:
            meta_key = json.dumps(meta_key, sort_keys=True)

        session = self._new_session()
        try:
            text_query = session.query(Text)

            if external_key is not None:
                text = text_query.filter(
                    Text.external_key == external_key.str,
                    Text.external_key_type == external_key.type).options(
                    lazyload('results'),
                    load_only('id', 'text', 'meta')).first()

                if text is not None:
                    if text.text != text_key:
                        raise falcon.HTTPConflict(
                            f"mismatch in stored text data for external key '{external_key.raw}'")

                    if text.meta != meta_key:
                        raise falcon.HTTPConflict(
                            f"mismatch in stored meta data for external key '{external_key.raw}'")

            elif text_key is not None:
                text = text_query.filter(
                    Text.text_hash_code == text_hash_code(text_key)).filter(
                    Text.text == text_key, Text.meta == meta_key).options(
                    load_only('id')).first().first()

            else:
                resp.status = falcon.HTTP_422
                return

            if text is None:
                new_text_guid = text_guid()

                if external_key is None:
                    external_key = new_text_guid

                if text_key is None:
                    raise falcon.HTTPInvalidParam(
                        "media", "missing text")

                text = Text(
                    guid=new_text_guid,
                    external_key=external_key.str,
                    external_key_type=external_key.type,
                    text=text_key,
                    text_hash_code=text_hash_code(text_key),
                    meta=meta_key)

                session.add(text)
                session.commit()
                session.refresh(text)

            resp.status = falcon.HTTP_200
            resp.text = json.dumps({
                'id': text.id
            })
        finally:
            session.close()


class ResultsResource:
    def __init__(self, new_session):
        self._new_session = new_session

    def on_get(self, req, resp, tagger_id, text_id):
        fields = req.params.get("fields")

        session = self._new_session()
        try:
            result = session.query(Result).filter(
                Result.tagger_id == tagger_id, Result.text_id == text_id).first()

            if result is None:
                resp.status = falcon.HTTP_404
                return

            data_acc = {
                'id': lambda: result.id,
                'status': lambda: result.status.name,
                'data': lambda: result.data,
                'time_created': lambda: result.time_created.isoformat()
            }

            if fields is not None:
                data = {}
                for f in fields.split(","):
                    k = f.strip()
                    if k not in data_acc:
                        raise falcon.HTTPInvalidParam(
                            "fields", f"illegal field {k}")
                    data[k] = data_acc[k]()
            else:
                data = dict((k, data_acc[k]()) for k in data_acc.keys())

            resp.status = falcon.HTTP_200
            resp.text = json.dumps(data)

        finally:
            session.close()

    def on_post(self, req, resp, tagger_id, text_id):
        result_data = req.media

        session = self._new_session()
        try:
            if session.query(Result).filter(
                    Result.tagger_id == tagger_id, Result.text_id == text_id).count() > 0:
                raise falcon.HTTPConflict(
                    f"Result for tagger {tagger_id}, text {text_id} is already in db.")

            tagger = session.query(Tagger).filter(Tagger.id == tagger_id).first()
            text = session.query(Text).filter(Text.id == text_id).first()

            result = json_to_result(
                tagger=tagger,
                text=text,
                status=ResultStatus[result_data['status']],
                json_data=result_data['data'])

            vectors = result_data.get('vectors')
            if vectors is not None:
                dtype = vectors['dtype']
                for k, v in vectors['data'].items():
                    x_vectors = [Vector(index=i, data=bytes.fromhex(x)) for i, x in enumerate(v)]
                    result.vectors.append(Vectors(name=k, dtype=dtype, vectors=x_vectors))

            session.add(result)
            session.commit()
            session.refresh(result)

            resp.status = falcon.HTTP_200
            resp.text = json.dumps({'id': result.id})

        finally:
            session.close()



@click.command()
@click.argument('path', type=click.Path(exists=False))
@click.option('--port', default=8000, help='Port to serve on.')
@click.option('--user', default="user", help='Username for basic auth.')
@click.option('--password', required=True, help='Password for basic auth.')
def run(path, port, user, password):
    """Run a server on the given carenero archive."""

    info = ArchiveInfo(path, engine='carenero')
    new_session = create_session_factory(info.base_path)

    auth_backend = BasicAuthBackend(functools.partial(user_loader, config={
        'user': user,
        'password': password
    }))
    auth_middleware = AuthMiddleware(auth_backend)
    app = falcon.App(middleware=[auth_middleware])

    app.add_route('/ping', PingResource())
    app.add_route('/taggers', TaggersResource(new_session))
    app.add_route('/taggers/{tagger_id:int}', TaggersByIdResource(new_session))
    app.add_route('/texts', TextsResource(new_session))
    app.add_route('/taggers/{tagger_id:int}/texts/{text_id:int}/results', ResultsResource(new_session))

    with make_server('', port, app) as httpd:
        print(f'Serving on port {port}...')

        # Serve until process is killed
        httpd.serve_forever()


if __name__ == '__main__':
    run()
