#Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################

import torch
import numpy as np 

#Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal(tensor, scale, zero, min_max, axis=0, device='cuda', opt_params={'lp_norm':0.7, 'beta':1e1, 'kappa':1.01, 'iters':20}, verbose=False):
	lp_norm, beta, kappa, iters = opt_params['lp_norm'], opt_params['beta'], opt_params['kappa'], opt_params['iters']

	dtype  = torch.float16 if (device=='cuda') else torch.float32
	W_f    = tensor.to(dtype).to(device)
	scale  = scale.to(dtype).to(device)
	zero   = zero.to(dtype).to(device)

	if(lp_norm==1):
		shrink_op = lambda x, beta: torch.sign(x)*torch.nn.functional.relu(torch.abs(x) - 1./beta) 
	else:
		shrink_op = lambda x, beta,p=lp_norm: torch.sign(x)*torch.nn.functional.relu(torch.abs(x) - (1./beta)*torch.pow(torch.abs(x), p-1))

	best_error = 1e4
	for i in range(iters):
		W_q   = torch.round(W_f*scale + zero).clamp(min_max[0], min_max[1])
		W_r   = (W_q - zero)/scale
		W_e   = shrink_op(W_f - W_r, beta)
		zero  = torch.mean(W_q - (W_f - W_e)*scale, axis=axis, keepdim=True)
		beta *= kappa

		current_error = float(torch.abs(W_f - W_r).mean())
		if(verbose): 
			print(i, np.round(current_error, 6))
		if(current_error < best_error):
			best_error = current_error
		else:
			break

	scale = scale.to(tensor.device)
	zero  = zero.to(tensor.device)
	del W_f, W_q, W_r, W_e
	torch.cuda.empty_cache()

	return scale, zero 

#SGD solver  || W - dequantize(quantize(W))||_1 (p=1 only)
def optimize_weights_autograd(tensor, scale, zero, min_max, axis=0, device='cuda', opt_params={'lr':2e-3, 'iters':2500}, verbose=False): 
	W_f             = tensor.to(device) 
	params          = {}
	params['scale'] = torch.nn.Parameter(scale.float().to(device), requires_grad=True)
	params['zero']  = torch.nn.Parameter(zero.float().to(device),  requires_grad=True)
	optimizer       = torch.optim.AdamW([params[k] for k in params], lr=opt_params['lr'], betas=(0.9, 0.99), eps=1e-06, weight_decay=0.)

	def _loss_fct(output, target):
		return torch.mean(torch.abs(target - output)) #L1

	def _fake_quant():
		#Quantize
		W_q = torch.round(W_f*params['scale'] + params['zero']).clamp(min_max[0], min_max[1])
		#Dequantize
		W_r = (W_q - params['zero'])/params['scale'] 
		return W_r

	with torch.no_grad():
		_init_loss = _loss_fct(_fake_quant(), W_f).item()

	def _step():
		optimizer.zero_grad()
		loss    = _loss_fct(_fake_quant(), W_f)
		loss.backward()
		optimizer.step()
		return  np.round(loss.item(), 10)

	for i in range(opt_params['iters']):
		l = _step()
		if(verbose and (i%100)==0): print(i, l)
		
	with torch.no_grad():
		_final_loss = _loss_fct(_fake_quant(), W_f).item()

	if(_final_loss<_init_loss):
		for k in params: params[k] = params[k].data.detach().to(tensor.device)
	else:
		if(verbose): print('optimization failed...')
		params = {'scale':scale, 'zero':zero}

	del W_f	
	torch.cuda.empty_cache()
	return params['scale'], params['zero']
