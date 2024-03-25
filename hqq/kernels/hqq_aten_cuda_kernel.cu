#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
inline  unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
#define BLOCK_SIZE 256 //~256 
#define SHARED_SIZE 512 //~512

//Custom Dispatcher to support Float, Half, Bfloat16 since the in Aten doens't support bfp16: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h#L248
#define AT_DISPATCHER_CASE(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)  \

#define AT_DISPATCHER(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCHER_CASE(__VA_ARGS__))

using namespace at;

/*******************************************************************************************************************************************/
/************* 8-bit *************/
/*******************************************************************************************************************************************/
//Simple
template <typename scalar_t>
__global__ void dequantize_8bit_u8_kernel(unsigned char* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int n = h*w;
	if(i>=n) return;

	int j   = i % w;
	W_r[i]  = (scalar_t(Wq_packed[i]) - zero[j])*scale[j];  
}


torch::Tensor dequantize_8bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero)
{
	CHECK_INPUT(Wq_packed);
	CHECK_INPUT(scale);
	CHECK_INPUT(zero);

	int r      = 1; //number of elements packed
	int h      = Wq_packed.size(0);
	int w      = Wq_packed.size(1);
	int n      = h*w; //num rows as a packed tensor

	auto dev   = Wq_packed.device();
	auto dtype = scale.dtype();
	auto W_r   = torch::empty({r*h, w}, torch::TensorOptions().dtype(dtype).device(dev)); 

	int blocks = cdiv(n, BLOCK_SIZE);

	AT_DISPATCHER(W_r.type(), "dequantize_8bit_u8", ([&] {
		dequantize_8bit_u8_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), 
																	scale.data_ptr<scalar_t>(), 
																	zero.data_ptr<scalar_t>(), 
																	W_r.data_ptr<scalar_t>(), 
																	h, w);
	}));


	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();

	return W_r;
}

/*******************************************************************************************************************************************/
/************* 4-bit *************/
/*******************************************************************************************************************************************/

//Simple
__global__ void unpack_4bit_u8_kernel(unsigned char* Wq_packed, unsigned char* Wq_unpacked, int n) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n) return;

	Wq_unpacked[i]     = (Wq_packed[i] & 0xF0) >> 4;  //First chunk
	Wq_unpacked[i + n] = (Wq_packed[i] & 0x0F);       //Second chunk	
}

torch::Tensor unpack_4bit_u8(torch::Tensor &Wq_packed)
{
	CHECK_INPUT(Wq_packed);

	int r = 2; //number of elements packed
	int h = Wq_packed.size(0);
	int w = Wq_packed.size(1);
	int n = h*w; //num rows as a packed tensor

	auto Wq_unpacked = torch::empty({h*r, w}, Wq_packed.options()); 

	int blocks = cdiv(n, BLOCK_SIZE);
	unpack_4bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), Wq_unpacked.data_ptr<unsigned char>(), n);	

	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();
	
	return Wq_unpacked;
}

//Simple
template <typename scalar_t>
__global__ void dequantize_4bit_u8_kernel(unsigned char* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int n = h*w;
	if(i>=n) return;

	int j      = i % w;
	W_r[i]     = (scalar_t((Wq_packed[i] & 0xF0) >> 4) - zero[j])*scale[j];  //First chunk
	W_r[i + n] = (scalar_t((Wq_packed[i] & 0x0F))      - zero[j])*scale[j];  //Second chunk
}

torch::Tensor dequantize_4bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero)
{
	CHECK_INPUT(Wq_packed);
	CHECK_INPUT(scale);
	CHECK_INPUT(zero);

	int r      = 2; //number of elements packed
	int h      = Wq_packed.size(0);
	int w      = Wq_packed.size(1);
	int n      = h*w; //num rows as a packed tensor

	auto dev   = Wq_packed.device();
	auto dtype = scale.dtype();
	auto W_r   = torch::empty({r*h, w}, torch::TensorOptions().dtype(dtype).device(dev)); 

	int blocks = cdiv(n, BLOCK_SIZE);

	AT_DISPATCHER(W_r.type(), "dequantize_4bit_u8", ([&] {
		dequantize_4bit_u8_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), 
																	scale.data_ptr<scalar_t>(), 
																	zero.data_ptr<scalar_t>(), 
																	W_r.data_ptr<scalar_t>(), 
																	h, w);
	}));


	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();

	return W_r;

}

/*******************************************************************************************************************************************/
/************* 2-bit *************/
/*******************************************************************************************************************************************/

//Simple
__global__ void unpack_2bit_u8_kernel(unsigned char* Wq_packed, unsigned char* Wq_unpacked, int n) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n) return;

	Wq_unpacked[i]       = (Wq_packed[i] & 0xC0) >> 6;  //1st chunk
	Wq_unpacked[i + n]   = (Wq_packed[i] & 0x30) >> 4;  //2nd chunk
	Wq_unpacked[i + n*2] = (Wq_packed[i] & 0x0C) >> 2;  //3rd chunk	
	Wq_unpacked[i + n*3] = (Wq_packed[i] & 0x03);       //4th chunk	
}


torch::Tensor unpack_2bit_u8(torch::Tensor &Wq_packed)
{
	CHECK_INPUT(Wq_packed);

	int r = 4; //number of elements packed
	int h = Wq_packed.size(0);
	int w = Wq_packed.size(1);
	int n = h*w; //num rows as a packed tensor

	auto Wq_unpacked = torch::empty({h*r, w}, Wq_packed.options()); 

	int blocks = cdiv(n, BLOCK_SIZE);
	unpack_2bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), Wq_unpacked.data_ptr<unsigned char>(), n);	

	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();
	
	return Wq_unpacked;
}

//Simple
template <typename scalar_t>
__global__ void dequantize_2bit_u8_kernel(unsigned char* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int n = h*w;
	if(i>=n) return;

	int j        = i % w;
	W_r[i]       = (scalar_t((Wq_packed[i] & 0xC0) >> 6) - zero[j])*scale[j];  //1st chunk
	W_r[i + n]   = (scalar_t((Wq_packed[i] & 0x30) >> 4) - zero[j])*scale[j];  //2nd chunk
	W_r[i + n*2] = (scalar_t((Wq_packed[i] & 0x0C) >> 2) - zero[j])*scale[j];  //3rd chunk	
	W_r[i + n*3] = (scalar_t((Wq_packed[i] & 0x03))      - zero[j])*scale[j];  //4th chunk	
}


// //Shared
// template <typename scalar_t>
// __global__ void dequantize_2bit_u8_kernel(unsigned char* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
// 	int i = blockIdx.x*blockDim.x + threadIdx.x;
// 	int n = h*w;
// 	int s = threadIdx.x;

// 	if(i>=n) return;

// 	__shared__ unsigned char shared[BLOCK_SIZE];
// 	__shared__ scalar_t shared_meta[BLOCK_SIZE][2];
	
// 	int j             = i % w;
// 	shared[s]         = Wq_packed[i];
// 	shared_meta[s][0] = zero[j];
// 	shared_meta[s][1] = scale[j];
// 	__syncthreads();


// 	W_r[i]       = (scalar_t((shared[s] & 0xC0) >> 6) - shared_meta[s][0])*shared_meta[s][1];  //1st chunk
// 	W_r[i + n]   = (scalar_t((shared[s] & 0x30) >> 4) - shared_meta[s][0])*shared_meta[s][1];  //2nd chunk
// 	W_r[i + n*2] = (scalar_t((shared[s] & 0x0C) >> 2) - shared_meta[s][0])*shared_meta[s][1];  //3rd chunk	
// 	W_r[i + n*3] = (scalar_t((shared[s] & 0x03))      - shared_meta[s][0])*shared_meta[s][1];  //4th chunk	
// }



torch::Tensor dequantize_2bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero)
{
	CHECK_INPUT(Wq_packed);
	CHECK_INPUT(scale);
	CHECK_INPUT(zero);

	int r      = 4; //number of elements packed
	int h      = Wq_packed.size(0);
	int w      = Wq_packed.size(1);
	int n      = h*w; //num rows as a packed tensor

	auto dev   = Wq_packed.device();
	auto dtype = scale.dtype();
	auto W_r   = torch::empty({r*h, w}, torch::TensorOptions().dtype(dtype).device(dev)); 

	int blocks = cdiv(n, BLOCK_SIZE);

	AT_DISPATCHER(W_r.type(), "dequantize_2bit_u8", ([&] {
		dequantize_2bit_u8_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), 
																	scale.data_ptr<scalar_t>(), 
																	zero.data_ptr<scalar_t>(), 
																	W_r.data_ptr<scalar_t>(), 
																	h, w);
	}));


	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();

	return W_r;
}

/*******************************************************************************************************************************************/
/************* 1-bit *************/
/*******************************************************************************************************************************************/

//Simple
__global__ void unpack_1bit_u8_kernel(unsigned char* Wq_packed, unsigned char* Wq_unpacked, int n) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n) return;

	Wq_unpacked[i]       = (Wq_packed[i] & 0x80) >> 7;  //1st chunk
	Wq_unpacked[i + n]   = (Wq_packed[i] & 0x40) >> 6;  //2nd chunk
	Wq_unpacked[i + n*2] = (Wq_packed[i] & 0x20) >> 5;  //3rd chunk	
	Wq_unpacked[i + n*3] = (Wq_packed[i] & 0x10) >> 4;  //4th chunk	
	Wq_unpacked[i + n*4] = (Wq_packed[i] & 0x08) >> 3;  //5th chunk	
	Wq_unpacked[i + n*5] = (Wq_packed[i] & 0x04) >> 2;  //6th chunk	
	Wq_unpacked[i + n*6] = (Wq_packed[i] & 0x02) >> 1;  //7th chunk	
	Wq_unpacked[i + n*7] = (Wq_packed[i] & 0x01);       //8th chunk	
}

torch::Tensor unpack_1bit_u8(torch::Tensor &Wq_packed)
{
	CHECK_INPUT(Wq_packed);

	int r = 8; //number of elements packed
	int h = Wq_packed.size(0);
	int w = Wq_packed.size(1);
	int n = h*w; //num rows as a packed tensor

	auto Wq_unpacked = torch::empty({h*r, w}, Wq_packed.options()); 

	int blocks = cdiv(n, BLOCK_SIZE);
	unpack_1bit_u8_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), Wq_unpacked.data_ptr<unsigned char>(), n);	

	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();
	
	return Wq_unpacked;
}

//Simple
template <typename scalar_t>
__global__ void dequantize_1bit_u8_kernel(unsigned char* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int n = h*w;
	if(i>=n) return;

	int j        = i % w;
	W_r[i]       = (scalar_t((Wq_packed[i] & 0x80) >> 7) - zero[j])*scale[j];  //1st chunk
	W_r[i + n]   = (scalar_t((Wq_packed[i] & 0x40) >> 6) - zero[j])*scale[j];  //2nd chunk
	W_r[i + n*2] = (scalar_t((Wq_packed[i] & 0x20) >> 5) - zero[j])*scale[j];  //3rd chunk	
	W_r[i + n*3] = (scalar_t((Wq_packed[i] & 0x10) >> 4) - zero[j])*scale[j];  //4th chunk	
	W_r[i + n*4] = (scalar_t((Wq_packed[i] & 0x08) >> 3) - zero[j])*scale[j];  //5th chunk	
	W_r[i + n*5] = (scalar_t((Wq_packed[i] & 0x04) >> 2) - zero[j])*scale[j];  //6th chunk	
	W_r[i + n*6] = (scalar_t((Wq_packed[i] & 0x02) >> 1) - zero[j])*scale[j];  //7th chunk	
	W_r[i + n*7] = (scalar_t((Wq_packed[i] & 0x01))      - zero[j])*scale[j];  //8th chunk	
}

// //Shared
// template <typename scalar_t>
// __global__ void dequantize_1bit_u8_kernel(unsigned char* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
// 	int i = blockIdx.x*blockDim.x + threadIdx.x;
// 	int s = threadIdx.x;
// 	int n = h*w;
// 	if(i>=n) return;

// 	__shared__ unsigned char shared[BLOCK_SIZE];
// 	__shared__ scalar_t shared_meta[BLOCK_SIZE][2];
	
// 	int j             = i % w;
// 	shared[s]         = Wq_packed[i];
// 	shared_meta[s][0] = zero[j];
// 	shared_meta[s][1] = scale[j];
// 	__syncthreads();

// 	W_r[i]       = (scalar_t((shared[s] & 0x80) >> 7) - shared_meta[s][0])*shared_meta[s][1]; //1st chunk
// 	W_r[i + n]   = (scalar_t((shared[s] & 0x40) >> 6) - shared_meta[s][0])*shared_meta[s][1]; //2nd chunk
// 	W_r[i + n*2] = (scalar_t((shared[s] & 0x20) >> 5) - shared_meta[s][0])*shared_meta[s][1]; //3rd chunk	
// 	W_r[i + n*3] = (scalar_t((shared[s] & 0x10) >> 4) - shared_meta[s][0])*shared_meta[s][1]; //4th chunk	
// 	W_r[i + n*4] = (scalar_t((shared[s] & 0x08) >> 3) - shared_meta[s][0])*shared_meta[s][1]; //5th chunk	
// 	W_r[i + n*5] = (scalar_t((shared[s] & 0x04) >> 2) - shared_meta[s][0])*shared_meta[s][1]; //6th chunk	
// 	W_r[i + n*6] = (scalar_t((shared[s] & 0x02) >> 1) - shared_meta[s][0])*shared_meta[s][1]; //7th chunk	
// 	W_r[i + n*7] = (scalar_t((shared[s] & 0x01))      - shared_meta[s][0])*shared_meta[s][1]; //8th chunk	
// }


torch::Tensor dequantize_1bit_u8(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero)
{
	CHECK_INPUT(Wq_packed);
	CHECK_INPUT(scale);
	CHECK_INPUT(zero);

	int r      = 8; //number of elements packed
	int h      = Wq_packed.size(0);
	int w      = Wq_packed.size(1);
	int n      = h*w; //num rows as a packed tensor

	auto dev   = Wq_packed.device();
	auto dtype = scale.dtype();
	auto W_r   = torch::empty({r*h, w}, torch::TensorOptions().dtype(dtype).device(dev)); 

	int blocks = cdiv(n, BLOCK_SIZE);

	AT_DISPATCHER(W_r.type(), "dequantize_1bit_u8", ([&] {
		dequantize_1bit_u8_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<unsigned char>(), 
																	scale.data_ptr<scalar_t>(), 
																	zero.data_ptr<scalar_t>(), 
																	W_r.data_ptr<scalar_t>(), 
																	h, w);
	}));


	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();

	return W_r;
}



/*******************************************************************************************************************************************/
/************* 3-bit *************/
/*******************************************************************************************************************************************/

//Simple
__global__ void unpack_3bit_32_kernel(int32_t* Wq_packed, unsigned char* Wq_unpacked, int n) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n) return;

	Wq_unpacked[i]       = (Wq_packed[i] & 0x38000000) >> 27;  //1st chunk
	Wq_unpacked[i + n]   = (Wq_packed[i] & 0x07000000) >> 24;  //2nd chunk
	Wq_unpacked[i + n*2] = (Wq_packed[i] & 0x00E00000) >> 21;  //3rd chunk	
	Wq_unpacked[i + n*3] = (Wq_packed[i] & 0x001C0000) >> 18;  //4th chunk	
	Wq_unpacked[i + n*4] = (Wq_packed[i] & 0x00038000) >> 15;  //5th chunk	
	Wq_unpacked[i + n*5] = (Wq_packed[i] & 0x00007000) >> 12;  //6th chunk	
	Wq_unpacked[i + n*6] = (Wq_packed[i] & 0x00000E00) >> 9;   //7th chunk	
	Wq_unpacked[i + n*7] = (Wq_packed[i] & 0x000001C0) >> 6;   //8th chunk	
	Wq_unpacked[i + n*8] = (Wq_packed[i] & 0x00000038) >> 3;   //9th chunk	
	Wq_unpacked[i + n*9] = (Wq_packed[i] & 0x00000007);        //10th chunk	
}


torch::Tensor unpack_3bit_32(torch::Tensor &Wq_packed)
{
	CHECK_INPUT(Wq_packed);

	int r = 10; //number of elements packed
	int h = Wq_packed.size(0);
	int w = Wq_packed.size(1);
	int n = h*w; //num rows as a packed tensor

	auto dev   = Wq_packed.device();
	auto dtype = torch::kByte;
	auto Wq_unpacked = torch::empty({r*h, w}, torch::TensorOptions().dtype(dtype).device(dev)); 

	int blocks = cdiv(n, BLOCK_SIZE);
	unpack_3bit_32_kernel<<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<int32_t>(), Wq_unpacked.data_ptr<unsigned char>(), n);	

	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();
	
	return Wq_unpacked;
}

//Simple
template <typename scalar_t>
__global__ void dequantize_3bit_32_kernel(int32_t* Wq_packed, scalar_t* scale, scalar_t* zero, scalar_t* W_r, int h, int w) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int n = h*w;
	if(i>=n) return;

	int j        = i % w;
	W_r[i]       = (scalar_t((Wq_packed[i] & 0x38000000) >> 27) - zero[j])*scale[j];  //1st chunk
	W_r[i + n]   = (scalar_t((Wq_packed[i] & 0x07000000) >> 24) - zero[j])*scale[j];  //2nd chunk
	W_r[i + n*2] = (scalar_t((Wq_packed[i] & 0x00E00000) >> 21) - zero[j])*scale[j];  //3rd chunk	
	W_r[i + n*3] = (scalar_t((Wq_packed[i] & 0x001C0000) >> 18) - zero[j])*scale[j];  //4th chunk	
	W_r[i + n*4] = (scalar_t((Wq_packed[i] & 0x00038000) >> 15) - zero[j])*scale[j];  //5th chunk	
	W_r[i + n*5] = (scalar_t((Wq_packed[i] & 0x00007000) >> 12) - zero[j])*scale[j];  //6th chunk	
	W_r[i + n*6] = (scalar_t((Wq_packed[i] & 0x00000E00) >> 9)  - zero[j])*scale[j];  //7th chunk	
	W_r[i + n*7] = (scalar_t((Wq_packed[i] & 0x000001C0) >> 6)  - zero[j])*scale[j];  //8th chunk	
	W_r[i + n*8] = (scalar_t((Wq_packed[i] & 0x00000038) >> 3)  - zero[j])*scale[j];  //9th chunk	
	W_r[i + n*9] = (scalar_t((Wq_packed[i] & 0x00000007))       - zero[j])*scale[j];  //10th chunk	
}

torch::Tensor dequantize_3bit_32(torch::Tensor &Wq_packed, torch::Tensor &scale, torch::Tensor &zero)
{
	CHECK_INPUT(Wq_packed);
	CHECK_INPUT(scale);
	CHECK_INPUT(zero);

	int r      = 10; //number of elements packed 
	int h      = Wq_packed.size(0);
	int w      = Wq_packed.size(1);
	int n      = h*w; //num rows as a packed tensor

	auto dev   = Wq_packed.device();
	auto dtype = scale.dtype();
	auto W_r   = torch::empty({r*h, w}, torch::TensorOptions().dtype(dtype).device(dev)); 

	int blocks = cdiv(n, BLOCK_SIZE);

	AT_DISPATCHER(W_r.type(), "dequantize_3bit_32", ([&] {
		dequantize_3bit_32_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(Wq_packed.data_ptr<int32_t>(), 
																	scale.data_ptr<scalar_t>(), 
																	zero.data_ptr<scalar_t>(), 
																	W_r.data_ptr<scalar_t>(), 
																	h, w);
	}));


	C10_CUDA_KERNEL_LAUNCH_CHECK();
	//cudaDeviceSynchronize();

	return W_r;

}
