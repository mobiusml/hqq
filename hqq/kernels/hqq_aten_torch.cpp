#include <torch/extension.h>
#include <vector>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/script.h>

inline torch::Tensor unpack_4bit_u8(torch::Tensor &W_q)
{
	return torch::cat({(W_q & 0xF0).__rshift__(4), (W_q & 0x0F)}, 0);
}

inline torch::Tensor unpack_3bit_32(torch::Tensor &W_q) {
    return torch::cat(
    {
        ((W_q & 0x38000000).__rshift__(27)),
        ((W_q & 0x07000000).__rshift__(24)),
        ((W_q & 0x00E00000).__rshift__(21)),
        ((W_q & 0x001C0000).__rshift__(18)),
        ((W_q & 0x00038000).__rshift__(15)),
        ((W_q & 0x00007000).__rshift__(12)),
        ((W_q & 0x00000E00).__rshift__(9)),
        ((W_q & 0x000001C0).__rshift__(6)),
        ((W_q & 0x00000038).__rshift__(3)),
        ((W_q & 0x00000007))
    }, 0);
}

inline torch::Tensor unpack_2bit_u8(torch::Tensor &W_q)
{
	return torch::cat({(W_q & 0xC0).__rshift__(6), (W_q & 0x30).__rshift__(4), (W_q & 0x0C).__rshift__(2), W_q & 0x03}, 0);  
}

inline torch::Tensor unpack_1bit_u8(torch::Tensor &W_q) {
    return torch::cat(
    {
        ((W_q & 0x80).__rshift__(7)),
        ((W_q & 0x40).__rshift__(6)),
        ((W_q & 0x20).__rshift__(5)),
        ((W_q & 0x10).__rshift__(4)),
        ((W_q & 0x08).__rshift__(3)),
        ((W_q & 0x04).__rshift__(2)),
        ((W_q & 0x02).__rshift__(1)),
        ((W_q & 0x01))
    }, 0);
}


inline torch::Tensor dequantize(torch::Tensor &W_q, torch::Tensor &scale, torch::Tensor &zero, torch::IntArrayRef &shape, int group_size, int nbits, int axis, std::string packing)
{
	torch::Tensor W_q_p;

	//Unpack bits
	if(packing=="8bit_u8"){W_q_p = W_q;}
	if(packing=="4bit_u8"){W_q_p = unpack_4bit_u8(W_q);}
	if(packing=="3bit_32"){W_q_p = unpack_3bit_32(W_q);}
	if(packing=="2bit_u8"){W_q_p = unpack_2bit_u8(W_q);}
	if(packing=="1bit_u8"){W_q_p = unpack_1bit_u8(W_q);}

	//Check size: 
	if(group_size>0 && nbits==3)
	{	
		W_q_p   = W_q_p.slice(axis, 0, group_size); 
	}

	//linear op: ToDO: use fp32 for cpu
	W_q_p    = W_q_p.to(torch::kHalf);
	auto W_r = ((W_q_p - zero) * scale).reshape(shape);

	return W_r;
}


inline torch::Tensor forward_with_quant(torch::Tensor &x,   torch::Tensor &bias, 
							    	    torch::Tensor &W_q, torch::Tensor &W_s, torch::Tensor &W_z, torch::IntArrayRef &W_shape, int W_group_size, int W_nbits, int W_axis, std::string W_packing,
								        torch::Tensor &S_q, torch::Tensor &S_s, torch::Tensor &S_z, torch::IntArrayRef &S_shape, int S_group_size, int S_nbits, int S_axis, std::string S_packing,
								        torch::Tensor &Z_q, torch::Tensor &Z_s, torch::Tensor &Z_z, torch::IntArrayRef &Z_shape, int Z_group_size, int Z_nbits, int Z_axis, std::string Z_packing) 
{

	torch::Tensor W_s_tmp, W_z_tmp;

	if(S_q.numel()>0){
		W_s_tmp = dequantize(S_q, S_s, S_z, S_shape, S_group_size, S_nbits, S_axis, S_packing);
	}
	else {
		W_s_tmp = W_s;
	}
	
	if(Z_q.numel()>0){
		W_z_tmp = dequantize(Z_q, Z_s,  Z_z, Z_shape, Z_group_size, Z_nbits, Z_axis, Z_packing);
	}
	else {
		W_z_tmp = W_z;
	}

	
	auto W_est = dequantize(W_q, W_s_tmp, W_z_tmp, W_shape, W_group_size, W_nbits, W_axis, W_packing).transpose(0, 1);
	auto out   = torch::matmul(x, W_est);
	if(bias.numel()>0) {out += bias;}

	return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_with_quant", &forward_with_quant, "forward_with_quant");
  m.def("dequantize",         &dequantize,         "dequantize");
}
