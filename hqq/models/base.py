#Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch 
import gc, os 
from tqdm import tqdm 
from abc import abstractmethod

from huggingface_hub import snapshot_download
from ..core.quantize import HQQLinear

def cleanup():
	torch.cuda.empty_cache()
	gc.collect()

def fix_path(path):
	if(len(path)==0): return path
	return path + '/' if (path[-1]!='/') else path 

#Base patching class. Patching defines how nn.Linear and other layers are replaced via a patching function. 
class BasePatch():
	#Override these OR override the main patch_model() function
	############################################
	#This method iterates through layers of the model that are NOT nn.Linear and processes them via new_nodule = patch_fct(module, params)
	@classmethod
	def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
		pass

	#This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
	@classmethod
	def patch_linearlayers(cls, base_model, patch_fct, patch_params, verbose=True):
		pass 
	############################################
	#These tags are used to specfiy parameters of the patching in  patch_linearlayers()
	@classmethod
	def get_linear_tags(cls):
		return []
	
	#Autmatically name modules. This is very important to save/load the weights 
	@classmethod
	def autoname_modules(cls, model):
		for name, module in model.named_modules():
			module.name = name

	#Freeze all layers
	@classmethod
	def freeze_model(cls, model):
		for param in model.parameters():
			param.requires_grad = False
		try:
			for param in model.model.parameters():
				param.requires_grad = False
		except:
			pass

	#Main patching function
	@classmethod
	def patch_model(cls, model, patch_nonlinear_fct, patch_linear_fct, patch_params, verbose=True):
		model.eval()
		cls.freeze_model(model)
		cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
		cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
		cls.autoname_modules(model)
		cleanup()


class BaseHQQModel:
	#Override these
	############################################
	#This method creates and empty model based on the specfied architecture
	@abstractmethod
	def create_model(self):
		pass

	#This method saves the model architecture only without inculding the weights (for example to a config.json)
	@abstractmethod	
	def cache_model(cls, model, save_dir):
		pass
	############################################

	@classmethod
	def get_config_file(cls, save_dir):
		return fix_path(save_dir) + 'config.json'

	@classmethod
	def get_weight_file(cls, save_dir):
		return fix_path(save_dir) + 'qmodel.pt'    

	@classmethod
	def get_ignore_layers(cls, model):
		return []

	@classmethod
	def save_weights(cls, weights, save_dir):
		torch.save(weights, cls.get_weight_file(save_dir))

	@classmethod
	def load_weights(cls, save_dir):
		return torch.load(cls.get_weight_file(save_dir))

	@classmethod
	def quantize_model(cls, model, quant_config):
		#Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
		patch_params = dict([(k, quant_config) for k in cls.get_linear_tags()])

		#We replace the nn.Linear layers with HQQLinear
		def _patch_linear(linear_layer, quant_config):
			return HQQLinear(linear_layer, quant_config) if (quant_config is not None) else linear_layer

		cls.patch_model(model, lambda l: l.half().cuda(), _patch_linear, patch_params)

	@classmethod
	def save_quantized(cls, model, save_dir, verbose=False):
		#Save config
		cls.cache_model(model, save_dir)

		#Save weights
		weights     = {}
		ignore_keys = cls.get_ignore_layers(model)
		for name, module in model.named_modules():
			if(name in ignore_keys): continue
			try:
				state_dict = module.state_dict()
				if(len(state_dict)>0): 
					weights[name] = dict(state_dict)
			except Exception as error:
				if(verbose): 
					print('Skipping', name)

		cls.save_weights(weights, save_dir)

	@classmethod
	def try_snapshot_download(cls, save_dir_or_hub, cache_dir=''):
		save_dir = fix_path(cache_dir) + save_dir_or_hub

		if(os.path.exists(save_dir)==False):
			save_dir = snapshot_download(repo_id=save_dir_or_hub, cache_dir=cache_dir)
			save_dir = fix_path(save_dir)

		#Check 
		if(os.path.exists(cls.get_weight_file(save_dir))==False):
			raise Exception('Weight file missing. Check your cache directory.')
		if(os.path.exists(cls.get_config_file(save_dir))==False):
			raise Exception('Config file missing. Check your cache directory.')

		return save_dir

	@classmethod
	def from_quantized(cls, save_dir_or_hub, cache_dir=''):
		#Get directory path
		save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

		#Load model from config
		model = cls.create_model(save_dir)

		#Name the layers
		cls.autoname_modules(model) 

		#Load weights
		try:
			weights = cls.load_weights(save_dir)
		except Exception as error:
			print("Failed to load the weights", error)
			return
		
		#load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
		@torch.no_grad()
		def _load_module(module, params=None):
			if(module.name not in weights): 
				return module.half().cuda()

			state_dict = weights[module.name]
			if(('W_q' in state_dict) and ('meta' in state_dict)):
				module = HQQLinear(linear_layer=None, quant_config=None)
				module.load_state_dict(state_dict)
			else:
				for key in state_dict:
					setattr(module, key, torch.nn.Parameter(state_dict[key]))

			return module 

		cls.patch_model(model, _load_module, _load_module, dict([(k, None) for k in cls.get_linear_tags()]))
		
		return model



