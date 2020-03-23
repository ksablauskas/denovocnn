import unittest
import pysam
from denovonet.variants import SingleVariant, TrioVariant
from denovonet.settings import OVERHANG

genome='hs37d5.fa.gz'
child_bam='child.bam'
father_bam='father.bam'
mother_bam='mother.bam'

REREFERENCE_GENOME = pysam.FastaFile('hs37d5.fa.gz')

class TestSingleVariant(unittest.TestCase):

    def setUp(self):
        self.test_variant = SingleVariant('chr1', 1000000, 1000000, child_bam, REREFERENCE_GENOME)
        pass

    def tearDown(self):
        pass

    def test_region_start(self):
        self.assertEqual(self.test_variant.region_start, 1000000 - OVERHANG - 2)

        self.test_variant.start = 1000001
        self.assertEqual(self.test_variant.region_start, 1000001 - OVERHANG - 2)

    def test_region_end(self):
        self.assertEqual(self.test_variant.region_end, 1000000 + OVERHANG - 2)

        self.test_variant.start = 1000001
        self.assertEqual(self.test_variant.region_end, 1000001 + OVERHANG - 2)

