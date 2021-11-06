import wget
import logging

MODEL_FILES_URL = [
    "https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru/resolve/main/config.json",
    "https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru/resolve/main/pytorch_model.bin",
    "https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru/resolve/main/tokenizer_config.json",
    "https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru/resolve/main/tokenizer.json"
]

OUTPUT_DIR = "weights/"

def main():
    logger = logging.getLogger()
    logger.info(f"Loading to folder: {OUTPUT_DIR}")

    for url in MODEL_FILES_URL:
        logger.info(f"Loading from: {url}")
        wget.download(url, out=OUTPUT_DIR)


if __name__ == "__main__":
    main()