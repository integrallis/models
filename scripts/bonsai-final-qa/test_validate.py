import unittest
from validate import validate_indices, validate_header

class ProtocolTest(unittest.TestCase):
    def test_complete_indices(self):
        validate_indices([{'index':i} for i in range(200)],range(200))
    def test_missing_duplicate_and_extra_rejected(self):
        for rows in [[{'index':0}],[{'index':0},{'index':0}],[{'index':0},{'index':2}]]:
            with self.assertRaises(ValueError):validate_indices(rows,range(2))
    def test_changed_generation_rejected(self):
        p={'inputs':{'public-qa-cases.json':'hash'},'context':2048}
        with self.assertRaises(ValueError):validate_header({'caseSha256':'hash','context':2048,'generation':{'temperature':1}},p,models=True)

if __name__ == '__main__':unittest.main()
