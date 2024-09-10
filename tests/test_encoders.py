import unittest

class TestCharacterEncoder(unittest.TestCase):
    def test_characterencoder(self):
        raw_text = shakespeare.raw_text
        encoder = Characters(raw_text)
        
        encoded = encoder.encode()
        decoded = encoder.decode(encoded)
        self.assertEqual(str(raw_text), str(decoded))
        original = "Hello"
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        self.assertEqual(str(original), str(decoded))

"""Deal with stupid annoying relative imports 
(thanks to Paolo Rovelli's answer at
https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py"""
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        from datasets.datasets import shakespeare
        from preprocessing.encoders import Characters
    else:
        from ..datasets.datasets import shakespeare
        from ..preprocessing.encoders import Characters
    unittest.main()
