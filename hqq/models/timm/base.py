from ..base import BaseHQQModel
import os
import json
import timm


class BaseHQQTimmModel(BaseHQQModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir, kwargs):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as error:
            print(error)

        with open(cls.get_config_file(save_dir), "w") as file:
            json.dump(model.default_cfg, file)

    # Create empty model
    @classmethod
    def create_model(cls, save_dir):
        with open(cls.get_config_file(save_dir), "r") as file:
            config = json.load(file)
        model = timm.create_model(
            config["architecture"] + "." + config["tag"], pretrained=False
        )
        return model
