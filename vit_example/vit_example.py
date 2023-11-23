import numpy as np 
import timm, torch 

from hqq.core.quantize import hqq_base_quant_config
from hqq.models.vit import ViTHQQ

#Model 
#model_id = 'vit_base_patch32_clip_224.laion2b' #ViT-B-32
#model_id = 'vit_large_patch14_clip_224.laion2b'#ViT-L-14
model_id = 'vit_huge_patch14_clip_224.laion2b' #ViT-H-14
 
#Load model (on CPU)
model = timm.create_model(model_id, pretrained=True)

#Quantize settings
#quant_config = hqq_base_quant_config(nbits=8, group_size=128)
quant_config = hqq_base_quant_config(nbits=4, group_size=64)
#quant_config = hqq_base_quant_config(nbits=3, group_size=64)
#quant_config = hqq_base_quant_config(nbits=2, group_size=16, quant_scale=True)

#Quantize
ViTHQQ.quantize_model(model, quant_config=quant_config)

###############################################################
# #Save model
# save_dir = "repo/" + model_id
# ViTHQQ.save_quantized(model, save_dir=save_dir)

# #Load model
# model = ViTHQQ.from_quantized(save_dir) 
###############################################################

#Load reference model to compare with
model_ref = timm.create_model(model_id, pretrained=True)
model_ref = model_ref.half().cuda()
model_ref.eval();

#Pre-processing 
mean_clip = np.array([0.4815, 0.4578, 0.4082], 'float32')
std_clip  = np.array([0.2686, 0.2613, 0.2758], 'float32')
def normalize_images_clip(data_np_in, BCHW=True): 
	data_t = torch.from_numpy(data_np_in).float() if(type(data_np_in)==np.ndarray) else data_np_in.float()
	data_t = (data_t/255. - mean_clip)/std_clip
	data_t = data_t.swapaxes(2, 3).swapaxes(1, 2) if (BCHW) else data_t
	return data_t

###############################################################
#Compare the compressed model with the original 
x = np.random.rand(16, 224, 224, 3)
x = normalize_images_clip(x).half().cuda()

#Quantize
with torch.no_grad():
	y_q = model(x)
	y_q /= torch.norm(y_q, p=2, dim=-1, keepdim=True)

#Full-precision
with torch.no_grad():
	y_r = model_ref(x)
	y_r /= torch.norm(y_r, p=2, dim=-1, keepdim=True)

#We want the dot product to be as close as possible to 1 
print('Average dot-product score', float(torch.diag(torch.matmul(y_q, y_r.t())).mean())) #~0.998 (ViT-H-14 @4bit)
