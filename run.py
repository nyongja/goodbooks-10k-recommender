import os

import fire
from tqdm import tqdm

from models.XGBoost import XGBoost
import core.config as conf


class Run(object):
    def __init__(self, mode="train", user_id=""):
        self.path = conf.dataset_path
        self.model = XGBoost()
        self.mode = mode
        self.user_id = user_id
        
    def mode_run(self) :
        if self.mode == "train" :
            self.model.train()
        elif self.mode == "predict" :
            self.model.predict()
        elif self.mode == "recommend" :
            self.model.recommend(self.user_id)

if __name__ == "__main__":
    fire.Fire(Run)

