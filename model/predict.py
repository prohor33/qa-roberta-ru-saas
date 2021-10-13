import hydra
import logging
import os
from omegaconf import OmegaConf


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logger = logging.getLogger("tramsformers-ner")

    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")


if __name__ == "__main__":
    main()