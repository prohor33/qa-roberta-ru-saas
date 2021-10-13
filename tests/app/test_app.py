import io
import pytest
import json
from app import app_main


@pytest.fixture
def client():
    app_main.app.config["TESTING"] = True

    with app_main.app.test_client() as client:
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
    rv = client.post("/predict")

    assert rv.status == "400 BAD REQUEST"
    assert json.loads(rv.data) == {"errorMsg": "Form key 'requestParameters' is not set!"}


def test_predict(client):

    rv = client.post(
        "/predict",
        data=dict(
            requestParameters=json.dumps(dict(
                msgId="1",
                msgTm="2019:08:10",
                workId="23",
                context="some text",
                questions=["q0", "q1"]
            ))
        ),
        content_type='application/json'
    )

    assert rv.status == "200 OK"
    annwer = json.loads(rv.data)
    assert annwer["workId"] == "23"

    with open('data/test_output.json', 'r') as gold_file:
        gold = json.load(gold_file)
        assert annwer["modelResult"] == gold
