import unittest
import tempfile
import os

class TestUtils(unittest.TestCase):
    def test_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.make_temp_file(tmpdir, "cp-0005.weights.h5")
            self.make_temp_file(tmpdir, "cp-0010.weights.h5")
            self.make_temp_file(tmpdir, "cp-0100.weights.h5")
            self.make_temp_file(tmpdir, "cp-15100.weights.h5")
            self.make_temp_file(tmpdir, "cp-15-16000.weights.h5")
            self.make_temp_file(tmpdir, "invalid.weights.h5")
            self.make_temp_file(tmpdir, "invalid.txt")
            self.make_temp_file(tmpdir, "invalid")
            latest = latest_checkpoint(tmpdir)
            self.assertIn("16000", latest)

        with tempfile.TemporaryDirectory() as tmpdir:
            latest = latest_checkpoint(tmpdir)
            self.assertIsNone(latest)

    def make_temp_file(self, tmpdir, name):
        open(os.path.join(tmpdir, name), 'a').close()


"""Deal with stupid annoying relative imports 
(thanks to Paolo Rovelli's answer at
https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py"""
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        from utils.checkpoint import latest_checkpoint
    else:
        from ..utils.checkpoint import latest_checkpoint
    unittest.main()
