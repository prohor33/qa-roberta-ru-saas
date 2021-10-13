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
from model.predict import extract_from_pdf


app = Flask(__name__, template_folder="templates")


def get_log_message(work_id, msg_id, rq_id, msg):
    return f"[{work_id}, {msg_id}, {rq_id}] {msg}"


def check_completeness(files):
    if len(files) != 1:
        raise IndexError(f"Expected 1 file, got {len(files)}")


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

    if "requestParameters" not in request.form:
        return make_response(jsonify({
            "errorMsg": "Form key 'requestParameters' is not set!"
        }), 400)

    input_params = json.loads(request.form["requestParameters"])

    for param in ["msgId", "workId"]:
        if param not in input_params:
            return make_response(jsonify({
                "errorMsg": f"Form key requestParameters/'{param}' is not set!"
            }), 400)

    t_start = time()

    try:
        file_list = list(request.files.listvalues())[0]

        check_completeness(file_list)

        pdf_input = io.BufferedReader(file_list[0])

        model_result = extract_from_pdf(pdf_input)

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
    app.run(host=cfg.server.host, port=cfg.server.port, debug=True)

    global logger 
    logger = logging.getLogger("qa-roberta-ru-saas")

    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    app.run(host=cfg.server.host, port=cfg.server.port, threaded=False)

if __name__ == "__main__":
    main()

    
