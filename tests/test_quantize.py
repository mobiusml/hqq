import unittest
import torch
from hqq.core.quantize import Quantizer, HQQLinear, BaseQuantizeConfig, HQQBackend

class TestQuantizer(unittest.TestCase):

    def setUp(self) -> None:
        # set seed
        torch.manual_seed(42)
        self.m = torch.nn.Linear(16,128)
        self.w = self.m.weight.data
        return super().setUp()
    
    
    def test_quantizer(self):
        W_q, meta = Quantizer.quantize(self.w, round_zero=True, optimize=True, view_as_float=False)
        w_dq = Quantizer.dequantize(W_q, meta, view_as_float=False)
        norm1 = torch.norm(self.w - w_dq, p=0.7)
        
        W_q, meta = Quantizer.quantize(self.w, round_zero=True, optimize=True, view_as_float=True)
        w_dq = Quantizer.dequantize(W_q, meta, view_as_float=True)
        norm2 = torch.norm(self.w - w_dq, p=0.7)
        
        self.assertTrue(torch.equal(norm1, norm2))
        
        
    def test_hqq_linear(self):
        
        quant_configs = [
                 BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, offload_meta=False),
                 BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=False, offload_meta=False),
                 BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=True, offload_meta=False),
                 BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True, offload_meta=False),
                 BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True, offload_meta=True),
                 BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, offload_meta=True)
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
            mq = HQQLinear(self.m, quant_cfg, compute_dtype=torch.bfloat16, initialize=False)
            HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
            mq.initialize()
            w_dqs.append(mq.dequantize_aten())
        
        norm1 = torch.norm(self.w.cuda() - w_dqs[0], p=0.7)
        norm2 = torch.norm(self.w.cuda() - w_dqs[1], p=0.7)
        norm3 = torch.norm(self.w.cuda() - w_dqs[2], p=0.7)
        norm4 = torch.norm(self.w.cuda() - w_dqs[3], p=0.7)
        norm5 = torch.norm(self.w.cuda() - w_dqs[4], p=0.7)
        
        assert torch.isclose(norm1, norm2, rtol=1e-2)
        assert torch.isclose(norm1, norm3, rtol=1e-2)
        assert torch.isclose(norm1, norm4, rtol=1e-2)
        assert torch.isclose(norm1, norm5, rtol=1e-2)
    
if __name__ == '__main__':
    unittest.main()
