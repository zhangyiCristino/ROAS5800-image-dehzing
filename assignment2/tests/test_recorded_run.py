import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dehaze.recorded import verified_completed_summary


class RecordedRunReuse(unittest.TestCase):
    def temporary_directory(self):
        base=(ROOT/'build/test-temp').resolve()
        assert base.is_relative_to(ROOT.resolve())
        base.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base)

    def write_record(self,path):
        payload=b'unit-test output'
        rows=[{'id':str(i),'output_sha256':hashlib.sha256(payload).hexdigest()} for i in range(352)]
        record={'complete':True,'summary':{'count':352,'marker':'verified'},'rows':rows}
        (path/'results.json').write_text(json.dumps(record),encoding='utf-8')
        return payload

    def test_fresh_clone_measurements_require_real_inference(self):
        with self.temporary_directory() as tmp:
            path=Path(tmp)
            self.write_record(path)
            self.assertIsNone(verified_completed_summary(path))

    def test_all_present_hash_verified_images_can_be_reused(self):
        with self.temporary_directory() as tmp:
            path=Path(tmp);payload=self.write_record(path)
            for i in range(352):(path/f'{i}.png').write_bytes(payload)
            self.assertEqual(verified_completed_summary(path),{'count':352,'marker':'verified'})

    def test_corrupt_present_output_is_not_silently_overwritten(self):
        with self.temporary_directory() as tmp:
            path=Path(tmp);self.write_record(path)
            (path/'0.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                verified_completed_summary(path)


if __name__=='__main__':unittest.main()
