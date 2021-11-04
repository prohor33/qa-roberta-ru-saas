# QA Roberta Ru SaaS
Question answering on russian with XLMRobertaLarge as a service. Thanks for the model to [Alexander Kaigorodov](https://huggingface.co/AlexKay).

## Build image

```
sudo docker build . --tag qa-roberta-ru-saas
```

## Run on CPU and predict
```
sudo docker run --rm -p 8080:8080 --name qa-roberta-ru-saas qa-roberta-ru-saas

curl -H "Content-Type: application/json" --data @tests/app/data/test_input.json 0.0.0.0:8080/predict
```

## Run on GPU
Change `device` to `cuda:0` in config before docker build:
```
device: cuda:0
```
After build:
```
sudo docker run --rm --gpus 0 -p 8080:8080 --name qa-roberta-ru-saas qa-roberta-ru-saas
```


## To run tests:
```
pytest tests/
```

## To run app without docker container
```
PYTHONPATH=. python app/app_main.py
```

## TODO:

- [x] GPU/CPU support
- [x] Support of context longer than 512 bpe
- [ ] Predict on long context with sliding window
