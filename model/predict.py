from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typeguard import typechecked
import torch
from typing import List
from hydra.utils import to_absolute_path


@typechecked
def load_model(cfg, logger):
    model_path = cfg.model_name
    if cfg.load_local_weights:
        model_path = to_absolute_path(cfg.local_weights)
        
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.to(cfg.device)
    logger.info(f"Model is succesfully loaded!")
    return model, tokenizer


@typechecked
def predict_from_text(cfg, logger, model, tokenizer, text: str, questions: List[str]) -> List[dict]:

    model_result = []
    MAX_TEXT_CHAR_LENGTH = 50000
    MAX_QUESTION_BPE_LENGTH = 256
    MAX_MODEL_INPUT_LENGTH = 512

    if len(text) > MAX_TEXT_CHAR_LENGTH:
        raise ValueError(f"Text length exceed max text length. {len(text)} > {MAX_TEXT_CHAR_LENGTH}")

    for question in questions:
        inputs_q = tokenizer(question, add_special_tokens=False, return_tensors="pt")
        question_length = inputs_q['input_ids'].shape[1]
        if question_length > MAX_QUESTION_BPE_LENGTH:
            raise ValueError(f"Question length long exceed max question length. {question_length} > {MAX_QUESTION_BPE_LENGTH}")

        inputs_t = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        text_ids_original = inputs_t["input_ids"].tolist()[0]

        # In order to support text longer than 512 bpe (model context size), we will split to several contexts
        max_text_lenth = MAX_MODEL_INPUT_LENGTH - question_length - 4 # 0, [Q], 2, 2, [T], 2
        contexts = inputs_t['input_ids'][0].split(max_text_lenth)

        full_text_answer_start_scores = []
        full_text_answer_end_scores = []
        
        for context in contexts:
            start_t = torch.tensor([0])
            middle_t = torch.tensor([2, 2])
            end_t = torch.tensor([2])
            input_ids = torch.cat((start_t, inputs_q['input_ids'][0], middle_t, context, end_t))
            inputs = {
                'input_ids': torch.unsqueeze(input_ids, dim=0),
                'attention_mask': torch.unsqueeze(torch.ones(input_ids.shape[0], dtype=int), dim=0)
            }

            inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            context_start = question_length + 3 # 0, [Q], 2, 2, [T], 2
            answer_start_scores = outputs.start_logits.to("cpu").detach()[0][context_start:-1]
            answer_end_scores = outputs.end_logits.to("cpu").detach()[0][context_start:-1]

            full_text_answer_start_scores.append(answer_start_scores)
            full_text_answer_end_scores.append(answer_end_scores)

        answer_start_scores = torch.cat(full_text_answer_start_scores)
        answer_end_scores = torch.cat(full_text_answer_end_scores)

        assert answer_start_scores.shape[0] == len(text_ids_original)
        assert answer_end_scores.shape[0] == len(text_ids_original)

        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(text_ids_original[answer_start:answer_end]))
        model_result.append(
            {
                "question": question,
                "answer": answer
            }
        )
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")

    return model_result
