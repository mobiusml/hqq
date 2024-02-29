import unittest
import torch
from hqq.core.quantize import Quantizer, HQQLinear, BaseQuantizeConfig, HQQBackend

# python -m unittest tests.test_quantize.TestQuantizer.test_quantizer
class TestQuantizer(unittest.TestCase):

    def setUp(self) -> None:
        # set seed
        torch.manual_seed(42)
        self.m = torch.nn.Linear(16,128)
        self.w = self.m.weight.data
        return super().setUp()
    
    
    def test_quantizer(self):    
        for compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for nbits in [8,4,3,2]:
                W_q, meta = Quantizer.quantize(self.w, nbits=nbits, round_zero=True, optimize=True, view_as_float=False)
                if nbits == 3:
                    assert W_q.dtype == torch.int32
                else:
                    assert W_q.dtype == torch.uint8
                w_dq = Quantizer.dequantize(W_q, meta, view_as_float=False)
                norm1 = torch.norm(self.w - w_dq, p=0.7)
                
                W_q, meta = Quantizer.quantize(self.w, nbits=nbits, round_zero=True, optimize=True, view_as_float=True, compute_dtype=compute_dtype)
                assert W_q.dtype == compute_dtype
                w_dq = Quantizer.dequantize(W_q, meta, view_as_float=True)
                norm2 = torch.norm(self.w - w_dq, p=0.7)
                
                self.assertTrue(torch.equal(norm1, norm2))
        
    def test_quantizer_cuda(self):
        w_cuda = self.w.cuda()
        W_q, meta = Quantizer.quantize(w_cuda, round_zero=True, optimize=True, view_as_float=False)
        w_dq = Quantizer.dequantize(W_q, meta, view_as_float=False)
        norm1 = torch.norm(w_cuda - w_dq, p=0.7)
        
        W_q, meta = Quantizer.quantize(w_cuda, round_zero=True, optimize=True, view_as_float=True)
        w_dq = Quantizer.dequantize(W_q, meta, view_as_float=True)
        norm2 = torch.norm(w_cuda - w_dq, p=0.7)
        
        self.assertTrue(torch.equal(norm1, norm2))        
         
    def test_hqq_linear(self):
        for compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            # FIXME: fails with 8-bit!
            # 8-bit norms: norm1=22.245920181274414, norm2=22.237092971801758, norm3=37.290225982666016, norm4=37.38299560546875, norm5=37.26346206665039
            for nbits in [4,3,2]:
                
                print(f"Testing nbits={nbits} and compute_dtype={compute_dtype}")
                
                quant_configs = [
                        BaseQuantizeConfig(nbits=nbits, group_size=64, quant_zero=False, quant_scale=False, offload_meta=False),
                        BaseQuantizeConfig(nbits=nbits, group_size=64, quant_zero=True, quant_scale=False, offload_meta=False),
                        BaseQuantizeConfig(nbits=nbits, group_size=64, quant_zero=False, quant_scale=True, offload_meta=False),
                        BaseQuantizeConfig(nbits=nbits, group_size=64, quant_zero=True, quant_scale=True, offload_meta=False),
                        BaseQuantizeConfig(nbits=nbits, group_size=64, quant_zero=True, quant_scale=True, offload_meta=True),
                        BaseQuantizeConfig(nbits=nbits, group_size=64, quant_zero=False, quant_scale=False, offload_meta=True)
                ]

                w_dqs = []
                for quant_cfg in quant_configs:
                    if quant_cfg['scale_quant_params']: 
                        quant_cfg['scale_quant_params']['group_size'] = 8
                    if quant_cfg['zero_quant_params']: 
                        if quant_cfg['offload_meta']:
                            quant_cfg['zero_quant_params']['group_size'] = 8
                            quant_cfg['zero_quant_params']['channel_wise'] = True
                        else:
                            quant_cfg['zero_quant_params']['group_size'] = None
                            quant_cfg['zero_quant_params']['channel_wise'] = False
                    mq = HQQLinear(self.m, quant_cfg, compute_dtype=compute_dtype, initialize=False)
                    HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
                    mq.initialize()
                    w_dqs.append(mq.dequantize_aten())
                
                norm1 = torch.norm(self.w.cuda() - w_dqs[0], p=0.7)
                norm2 = torch.norm(self.w.cuda() - w_dqs[1], p=0.7)
                norm3 = torch.norm(self.w.cuda() - w_dqs[2], p=0.7)
                norm4 = torch.norm(self.w.cuda() - w_dqs[3], p=0.7)
                norm5 = torch.norm(self.w.cuda() - w_dqs[4], p=0.7)
                norm6 = torch.norm(self.w.cuda() - w_dqs[5], p=0.7)
                
                print(f"norm1={norm1}, norm2={norm2}, norm3={norm3}, norm4={norm4}, norm5={norm5}, norm6={norm6}")
                
                assert torch.isclose(norm1, norm2, rtol=1e-2)
                assert torch.isclose(norm1, norm3, rtol=1e-2)
                assert torch.isclose(norm1, norm4, rtol=1e-2)
                assert torch.isclose(norm1, norm5, rtol=1e-2)
                assert torch.isclose(norm1, norm6, rtol=1e-2)
        
if __name__ == '__main__':
    unittest.main()
