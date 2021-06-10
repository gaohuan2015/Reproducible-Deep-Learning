import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from model.Factory import ModelFactory
from helper.Preprocess import ReadCSVDataForClass
logger = logging.getLogger(__name__)

@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    model = ModelFactory.CreateModel(cfg)
    epoch = cfg.trainer.max_epochs
    assert model is not None

if __name__ == "__main__":
    train()