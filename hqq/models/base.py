#Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch 
import gc, os
from tqdm import tqdm 
from abc import abstractmethod

from huggingface_hub import snapshot_download
from ..core.quantize import HQQLinear

#Defined what is qualified as "linear layer"
_LINEAR_LAYERS = [torch.nn.Linear]
_IGNORE_LINEAR = ["lm_head"]

#Cleanup GPU vram
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

#Make sure file paths end with '/'
def fix_path(path):
    if(len(path)==0): return path
    return path + '/' if (path[-1]!='/') else path 

# Finds the parent of a node module named "name"
def find_parent(model, name):
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent

# checks if a module is a leaf: doesn't have another module inside
def is_leaf_module(module):
    return len(module._modules) == 0

# Get the linear_tag from a modul name. For example: model.layers.31.self_attn.k_proj -> self_attn.k_proj
def name_to_linear_tag(name):
    return ".".join([n for n in name.split(".") if ((n not in ["model", "layers"]) and (not n.isnumeric()))])

# Get all linear tags available
def get_linear_tags_from_model(model, ignore):
    linear_tags = set()
    for name, module in model.named_modules():
        if ((type(module) in _LINEAR_LAYERS) and (name.split('.')[-1] not in ignore)):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)


#Base patching class. Patching defines how nn.Linear and other layers are replaced via a patching function. 
class BasePatch():
    #Override these OR override the main patch_model() function
    ############################################
    #This method iterates through layers of the model that are NOT nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if((type(module) not in _LINEAR_LAYERS) and (name not in ignore_tags)):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            setattr(find_parent(model, name), name.split('.')[-1], patch_fct(tmp_mapping[name]))

    #This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if((type(module) in _LINEAR_LAYERS) and (name not in ignore_tags)):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag  = name_to_linear_tag(name)
            patch_param = patch_params[linear_tag] if (linear_tag in patch_params) else None
            setattr(find_parent(model, name), name.split('.')[-1], patch_fct(tmp_mapping[name], patch_param))


    ############################################
    #These tags are used to specfiy parameters of the patching in patch_linearlayers()
    @classmethod 
    def set_auto_linear_tags(cls, model, ignore=_IGNORE_LINEAR):
        if(len(cls.get_linear_tags())==0):
            cls.linear_tags     = get_linear_tags_from_model(model, ignore=ignore)
            cls.get_linear_tags = lambda: cls.linear_tags 

    #Returns the current linear tags
    @classmethod
    def get_linear_tags(cls):
        return []

    @classmethod
    def get_ignore_layers(cls, model):
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module):
                layers.add(name)
        return list(layers)
    
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
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
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

    #Save weights to disk
    @classmethod
    def save_weights(cls, weights, save_dir):
        torch.save(weights, cls.get_weight_file(save_dir))

    #Load weights from disk
    @classmethod
    def load_weights(cls, save_dir, map_location=None):
        return torch.load(cls.get_weight_file(save_dir), map_location=map_location)

    #Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype=torch.float16, device='cuda'):
        #Set linear tags automatically
        cls.set_auto_linear_tags(model)

        #Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if(True in [(key in cls.get_linear_tags()) for key in quant_config.keys()]): 
            #If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in cls.get_linear_tags()}
            patch_params.update(quant_config)
        else:
            #Same quant_config for all layers
            patch_params =  {k: quant_config for k in cls.get_linear_tags()}

        #We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            if(quant_config is not None):
                out_module = HQQLinear(linear_layer, quant_config, compute_dtype=compute_dtype, device=device) 
            else:
                out_module = linear_layer.to(device=device, dtype=compute_dtype, non_blocking=True)
            return out_module

        cls.patch_model(model, lambda l: l.to(device=device, dtype=compute_dtype, non_blocking=True), _patch_linear, patch_params)

        #Set base class
        model.base_class = cls

        return model

    #Prepares model weights by iterating through modules. It might some parameters that are NOT modules like model.param1
    @classmethod
    def serialize_weights(cls, model, verbose=False):
        weights     = {}
        ignore_keys = cls.get_ignore_layers(model)
        for name, module in model.named_modules():
            if(name in ignore_keys): 
            	continue
            try:
                state_dict = module.state_dict()
                if(len(state_dict)>0): 
                    weights[name] = dict(state_dict)
            except Exception:
                if(verbose): 
                    print('Skipping', name)

        return weights

    #Main function to save a quantized model
    @classmethod
    def save_quantized(cls, model, save_dir, verbose=False):
        #Save config
        cls.cache_model(model, save_dir)

        #Serialization
        weights = cls.serialize_weights(model, verbose=verbose)

        #Save
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


    #This method is specfically designed in case we need to load some weights that are not part of any module
    @classmethod
    def post_module_load(cls, model, weights):
        pass

    #Main function to load an HQQ quantized model from either HF hub or locally
    @classmethod
    def from_quantized(cls, save_dir_or_hub, compute_dtype=torch.float16, device='cuda', cache_dir=''):
        #Get directory path
        save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

        #Load model from config
        model = cls.create_model(save_dir)

        #Name the layers
        cls.autoname_modules(model) 

        #Set linear tags automatically
        cls.set_auto_linear_tags(model)

        #Load weights
        try:
            weights = cls.load_weights(save_dir)
        except Exception:
            print("Failed to load the weights")
            return
        
        #load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
        @torch.no_grad()
        def _load_module(module, params=None):
            if(module.name not in weights): 
                return module.to(device=device, dtype=compute_dtype, non_blocking=True)

            state_dict = weights[module.name]
            if(('W_q' in state_dict) and ('meta' in state_dict)):
                module = HQQLinear(linear_layer=None, quant_config=None, compute_dtype=compute_dtype, device=device)
                module.load_state_dict(state_dict)
            else:
                for key in state_dict:
                    setattr(module, key, torch.nn.Parameter(state_dict[key].to(device=device, dtype=compute_dtype, non_blocking=True), requires_grad=False))

            return module 

        #Load modules
        cls.patch_model(model, _load_module, _load_module, {k: None for k in cls.get_linear_tags()})

        #Load other weights that are not part of any module
        cls.post_module_load(model, weights) 

        #Set base class
        model.base_class = cls
        
        return model



