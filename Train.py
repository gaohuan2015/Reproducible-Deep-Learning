import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    train()