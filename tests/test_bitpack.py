import unittest
import torch
from hqq.core.bitpack import BitPack
try:
    import hqq_aten
except:
    print('hqq_aten package not available')
    hqq_aten = None

# python -m unittest test_bitpack.py
class TestBitPack(unittest.TestCase):

    def setUp(self) -> None:
        # set seed
        torch.manual_seed(42)
        self.device  = 'cuda'
        self.repeats = 10
        self.shapes = [[32, 32], [128, 256], [256, 256], [512, 512], [1024, 1024], [2048, 2048], [4096, 4096], [8192, 8192], [8192, 4096], [8192, 128], [32, 4096]]
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_nbit_torch(self, nbits=None, pack_fct=None, unpack_fct=None, device='cuda'):
        if(nbits is None): 
            self.assertTrue(True)
            return

        for compute_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            for shape in self.shapes:
                for _ in range(self.repeats):
                    W   = torch.randint(0, 2**nbits, shape, device=device).contiguous()
                    W_r = unpack_fct(pack_fct(W), dtype=compute_dtype)                    
                    self.assertTrue(torch.equal(W, W_r[:len(W)]))


    def test_nbit_aten(self, nbits=None, pack_fct=None, unpack_fct=None, device='cuda'):
        if(nbits is None): 
            self.assertTrue(True)
            return

        for compute_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            for shape in self.shapes:
                for _ in range(self.repeats):
                    W   = torch.randint(0, 2**nbits, shape, device=device).contiguous()
                    W_r = unpack_fct(pack_fct(W)).to(compute_dtype)                     
                    self.assertTrue(torch.equal(W, W_r[:len(W)]))

    #8-bit
    def test_8bit_u8_torch(self):
        return self.test_nbit_torch(nbits=8, pack_fct=BitPack.pack_8bit_u8, unpack_fct=BitPack.unpack_8bit_u8, device=self.device)

    #4-bit
    def test_4bit_u8_torch(self):
        return self.test_nbit_torch(nbits=4, pack_fct=BitPack.pack_4bit_u8, unpack_fct=BitPack.unpack_4bit_u8, device=self.device)

    @unittest.skipIf(hqq_aten is None, "hqq_aten not available")
    def test_4bit_u8_aten(self):
        return self.test_nbit_aten(nbits=4, pack_fct=BitPack.pack_4bit_u8, unpack_fct=hqq_aten.unpack_4bit_u8, device=self.device)
    
    #2-bit
    def test_2bit_u8_torch(self):
        return self.test_nbit_torch(nbits=2, pack_fct=BitPack.pack_2bit_u8, unpack_fct=BitPack.unpack_2bit_u8, device=self.device)

    @unittest.skipIf(hqq_aten is None, "hqq_aten not available")
    def test_2bit_u8_aten(self):
        return self.test_nbit_aten(nbits=2, pack_fct=BitPack.pack_2bit_u8, unpack_fct=hqq_aten.unpack_2bit_u8, device=self.device)

    #1-bit
    def test_1bit_u8_torch(self):
        return self.test_nbit_torch(nbits=1, pack_fct=BitPack.pack_1bit_u8, unpack_fct=BitPack.unpack_1bit_u8, device=self.device)

    @unittest.skipIf(hqq_aten is None, "hqq_aten not available")
    def test_1bit_u8_aten(self):
        return self.test_nbit_aten(nbits=1, pack_fct=BitPack.pack_1bit_u8, unpack_fct=hqq_aten.unpack_1bit_u8, device=self.device)

    #3-bit
    def test_3bit_32_torch(self):
        return self.test_nbit_torch(nbits=3, pack_fct=BitPack.pack_3bit_32, unpack_fct=BitPack.unpack_3bit_32, device=self.device)

    @unittest.skipIf(hqq_aten is None, "hqq_aten not available")
    def test_3bit_32_aten(self):
        return self.test_nbit_aten(nbits=3, pack_fct=BitPack.pack_3bit_32, unpack_fct=hqq_aten.unpack_3bit_32, device=self.device)
                                    
if __name__ == '__main__':
    unittest.main()
