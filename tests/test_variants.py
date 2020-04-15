import unittest
import os
import pysam
from denovonet.variants import SingleVariant, TrioVariant
from denovonet.settings import OVERHANG

TEST_DATA = os.path.join('tests','data')

genome=os.path.join(TEST_DATA,'hla.fa')
child_bam=os.path.join(TEST_DATA,'NA12878.chr22.tiny.hla.bam')
father_bam=os.path.join(TEST_DATA,'NA12878.chr22.tiny.hla.bam')
mother_bam=os.path.join(TEST_DATA,'NA12878.chr22.tiny.hla.bam')

REREFERENCE_GENOME = pysam.FastaFile(genome)

class TestSingleVariant(unittest.TestCase):

    def setUp(self):
        self.test_variant = SingleVariant('HLA-DRB1*16:02:01', 6030, 6030, child_bam, REREFERENCE_GENOME)
        pass

    def tearDown(self):
        pass

    def test_region_start(self):
        self.assertEqual(self.test_variant.region_start, 6030 - OVERHANG - 2)

        self.test_variant.start = 6031
        self.assertEqual(self.test_variant.region_start, 6031 - OVERHANG - 2)

    def test_region_end(self):
        self.assertEqual(self.test_variant.region_end, 6030 + OVERHANG - 2)

        self.test_variant.start = 6031
        self.assertEqual(self.test_variant.region_end, 6031 + OVERHANG - 2)

