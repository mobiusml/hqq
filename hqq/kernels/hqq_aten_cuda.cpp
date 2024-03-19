#include <torch/extension.h>
#include <vector>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <cassert>

torch::Tensor dequantize_8bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero);

torch::Tensor unpack_4bit_u8(torch::Tensor &Wq_packed);
torch::Tensor dequantize_4bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero);

torch::Tensor unpack_3bit_32(torch::Tensor &Wq_packed);
torch::Tensor dequantize_3bit_32(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero);

torch::Tensor unpack_2bit_u8(torch::Tensor &Wq_packed);
torch::Tensor dequantize_2bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero);

torch::Tensor unpack_1bit_u8(torch::Tensor &Wq_packed);
torch::Tensor dequantize_1bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero);

/*	scale & zero should be unquantized and already on the target device. 
	All tensors should be contiguous !!
	Only axis=0 is supported since the CUDA kernels heavily rely on this logic. 
*/
inline torch::Tensor dequantize(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero, torch::IntArrayRef &shape, int group_size, int nbits, int axis, std::string packing)
{	
	//Only axis=0 supported
	assert(axis == 0 && "Only axis=0 is supported.");

	torch::Tensor W_r;

	//Unpack bits
	if(packing=="8bit_u8"){W_r = dequantize_8bit_u8(Wq_packed, scale, zero);}
	if(packing=="4bit_u8"){W_r = dequantize_4bit_u8(Wq_packed, scale, zero);}
	if(packing=="3bit_32"){W_r = dequantize_3bit_32(Wq_packed, scale, zero);}
	if(packing=="2bit_u8"){W_r = dequantize_2bit_u8(Wq_packed, scale, zero);}
	if(packing=="1bit_u8"){W_r = dequantize_1bit_u8(Wq_packed, scale, zero);}

	//Check size for 3 bits
	if(group_size>0 && nbits==3)
	{	
		W_r   = W_r.slice(axis, 0, group_size); 
	}

	return W_r.reshape(shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dequantize_8bit_u8", &dequantize_8bit_u8, "dequantize_8bit_u8");	

	m.def("unpack_4bit_u8",     &unpack_4bit_u8,     "unpack_4bit_u8");
	m.def("dequantize_4bit_u8", &dequantize_4bit_u8, "dequantize_4bit_u8");

	m.def("unpack_3bit_32",     &unpack_3bit_32,     "unpack_3bit_32");
	m.def("dequantize_3bit_32", &dequantize_3bit_32, "dequantize_3bit_32");

	m.def("unpack_2bit_u8",     &unpack_2bit_u8,     "unpack_2bit_u8");
	m.def("dequantize_2bit_u8", &dequantize_2bit_u8, "dequantize_2bit_u8");

	m.def("unpack_1bit_u8",     &unpack_1bit_u8,     "unpack_1bit_u8");
	m.def("dequantize_1bit_u8", &dequantize_1bit_u8, "dequantize_1bit_u8");

	m.def("dequantize",         &dequantize,         "dequantize");
}
