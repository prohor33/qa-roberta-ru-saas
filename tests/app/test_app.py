import io
import pytest
import json
from app.app_main import create_app
import os
from hydra.utils import get_original_cwd, to_absolute_path
from hydra import compose, initialize


@pytest.fixture
def client():
    print("Working directory : {}".format(os.getcwd()))
    
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health(client):
    rv = client.get("/health")
    assert rv.status == "200 OK"
    assert json.loads(rv.data) == {"health_status": "running"}


def test_version(client):
    rv = client.get("/version")
    assert rv.status == "200 OK"
    assert json.loads(rv.data) == {"versions": {"common": "1.0.0"}}


def test_predict_no_params(client):
    rv = client.post(
        "/predict",
        content_type='application/json'
    )

    assert rv.status == "400 BAD REQUEST"


def test_predict(client):

    with open(to_absolute_path('tests/app/data/test_input.json'), 'r') as input_file:
        input = json.load(input_file)

    rv = client.post(
        "/predict",
        json=input,
        content_type='application/json'
    )

    assert rv.status == "200 OK"
    answer = json.loads(rv.data)
    assert answer["workId"] == "23"

    with open(to_absolute_path('tests/app/data/test_output.json'), 'r') as gold_file:
        gold = json.load(gold_file)
        assert answer["modelResult"] == gold
