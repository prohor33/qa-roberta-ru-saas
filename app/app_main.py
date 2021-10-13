import io
import json
import hydra
from omegaconf import OmegaConf
import logging
import os
from time import time
from flask import Flask, jsonify, make_response, render_template, request, current_app
from app.utils import (
    create_ok_response, create_error_response
)
from model.predict import load_model, predict_from_text


app = Flask(__name__, template_folder="templates")


def get_log_message(work_id, msg_id, rq_id, msg):
    return f"[{work_id}, {msg_id}, {rq_id}] {msg}"


@app.route("/version", methods=["GET"])
def version():
    cfg = current_app.config["config"]
    version_data = {
        'common': cfg.common_version
    }
    return make_response(jsonify({"versions": version_data}), 200)


@app.route("/health", methods=["GET"])
def health():
    output_data = {
        'health_status': 'running'
    }
    return make_response(jsonify(output_data), 200)


@app.route("/predict", methods=["POST"])
def predict():
    """
    params: {
        "msgId":     <unique message id>
        "workId":    <unique request id, will be returned in answer>
        "msgTm":     <message time in format: %Y-%m-%dT%H:%M:%S.%fZ>
        "context":   <context which the model will extract answers from>
        "questions": <list of questions to answer>
    }
    """
    cfg = current_app.config["config"]

    if "requestParameters" not in request.form:
        return make_response(jsonify({
            "errorMsg": "Form key 'requestParameters' is not set!"
        }), 400)

    input_params = json.loads(request.form["requestParameters"])

    for param in ["msgId", "workId", "msgTm", "context", "questions"]:
        if param not in input_params:
            return make_response(jsonify({
                "errorMsg": f"Form key requestParameters/'{param}' is not set!"
            }), 400)

    t_start = time()

    try:
        text = input_params["context"]
        questions = input_params["questions"]
        model_result = predict_from_text(cfg, logger, model, tokenizer, text, questions)

    except Exception as e:
        output_data = create_error_response(
            msg_id=input_params["msgId"],
            work_id=input_params["workId"],
            error_msg=str(e)
        )
        return make_response(jsonify(output_data), 500)

    t_end = time()

    output_data = create_ok_response(
        msg_id=input_params["msgId"],
        work_id=input_params["workId"],
        model_result=model_result,
        model_time=t_end - t_start
    )

    return make_response(json.dumps(output_data, ensure_ascii=False), 200)


@app.route("/")
def index():
    return render_template("index.html")


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    app.config["config"] = cfg

    global logger 
    logger = logging.getLogger("qa-roberta-ru-saas")

    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    global model, tokenizer
    model, tokenizer = load_model(cfg, logger)

    app.run(host=cfg.server.host, port=cfg.server.port, threaded=False)

if __name__ == "__main__":
    main()

    
