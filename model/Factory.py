import hydra
from model.Models import TextCNN
from omegaconf import DictConfig

class ModelFactory(object):
    @staticmethod
    def CreateModel(cfg: DictConfig):
        if cfg.model.name.lower() == 'textcnn':
            return TextCNN(cfg)
        else:
            return None