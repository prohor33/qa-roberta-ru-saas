from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typeguard import typechecked
import torch
from typing import List


@typechecked
def load_model(cfg, logger):
    logger.info(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)
    model.to(cfg.device)
    logger.info(f"Model is succesfully loaded!")
    return model, tokenizer


@typechecked
def predict_from_text(cfg, logger, model, tokenizer, text: str, questions: List[str]) -> List[dict]:

    model_result = []

    for question in questions:
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits.to("cpu").detach()
        answer_end_scores = outputs.end_logits.to("cpu").detach()
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        model_result.append(
            {
                "question": question,
                "answer": answer
            }
        )
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")

    return model_result
