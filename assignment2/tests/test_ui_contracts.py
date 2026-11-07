import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import app

class UIContracts(unittest.TestCase):
    def test_reference_dimensions_rejected(self):
        with self.assertRaisesRegex(ValueError,'dimensions'):
            app.prepare_pair(np.zeros((512,640,3),dtype=np.uint8),np.zeros((410,512,3),dtype=np.uint8),'512 px')

    def test_resize_preserves_input_reference_alignment(self):
        rng=np.random.default_rng(7)
        image=rng.integers(0,256,size=(512,640,3),dtype=np.uint8)
        a,b=app.prepare_pair(image,image.copy(),'512 px')
        self.assertEqual(a.shape,(410,512,3))
        self.assertTrue(np.array_equal(a,b))

    def test_native_input_preserved(self):
        image=np.arange(512*640*3,dtype=np.uint8).reshape(512,640,3)
        a,_=app.prepare_pair(image,None,'Original size')
        self.assertTrue(np.array_equal(a,image))

    def test_extremely_narrow_input_not_distorted(self):
        with self.assertRaisesRegex(ValueError,'smaller than 32'):
            app.prepare_pair(np.zeros((64,3000,3),dtype=np.uint8),None,'256 px')

    def test_quality_without_reference_and_perfect_match(self):
        image=np.zeros((32,32,3),dtype=np.uint8)
        self.assertIsNone(app.quality(image,None)['psnr'])
        perfect=app.quality(image,image)
        self.assertTrue(np.isposinf(perfect['psnr']))
        self.assertEqual(perfect['ssim'],1.0)
        self.assertEqual(app.number(perfect['psnr']),'∞')

    def test_input_change_invalidates_outputs(self):
        previous=app.GENERATION['direct-test']
        result=app.invalidate(None)
        self.assertEqual(app.GENERATION['direct-test'],previous+1)
        self.assertEqual(result[:3],(None,None,None))
        self.assertEqual(result[3],[])
        self.assertFalse(result[-1]['visible'])
        self.assertIsNone(result[-1]['value'])

    def test_partial_benchmark_is_not_presented_as_complete(self):
        base=(ROOT/'build/test-temp').resolve()
        assert base.is_relative_to(ROOT.resolve())
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            path=Path(tmp)
            (path/'status.json').write_text(json.dumps({'method':'IPC-Dehaze','done':1,'total':352}),encoding='utf-8')
            with patch.object(app,'EXP',path):
                rows,choices,note=app.saved_status()
            self.assertEqual(rows,[])
            self.assertEqual(choices,[])
            self.assertIn('in progress',note)

if __name__=='__main__':unittest.main()
