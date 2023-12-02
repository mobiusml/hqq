from tqdm import tqdm 
import transformers
from accelerate import init_empty_weights

from ..base import BaseHQQModel
class BaseHQQHFModel(BaseHQQModel):
	#Save model architecture
	@classmethod
	def cache_model(cls, model, save_dir):
		model.config.save_pretrained(save_dir)
