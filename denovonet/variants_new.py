import pysam
import numpy as np
import cv2
from PIL import Image
import itertools

# Import Denovonet Tools
from denovonet.settings import OVERHANG, IMAGE_WIDTH, PLACEHOLDER_WIDTH, IMAGE_HEIGHT
from denovonet.encoders import baseEncoder, baseDecoder, VariantClassValue
from denovonet.colors import bcolors

def decode_base(base):
    if base == baseEncoder.A:
        return baseDecoder.A
    elif base == baseEncoder.C:
        return baseDecoder.C
    elif base == baseEncoder.T:
        return baseDecoder.T
    elif base == baseEncoder.G:
        return baseDecoder.G
    elif base == baseEncoder.IN_A:
        return baseDecoder.IN_A
    elif base == baseEncoder.IN_C:
        return baseDecoder.IN_C
    elif base == baseEncoder.IN_T:
        return baseDecoder.IN_T
    elif base == baseEncoder.IN_G:
        return baseDecoder.IN_G
    elif base == baseEncoder.DEL:
        return baseDecoder.DEL
    elif base == baseEncoder.EMPTY:
        return baseDecoder.EMPTY

class SingleVariant():

    def __init__(self, chromosome, start, end, bam_path, REREFERENCE_GENOME):
        # Variant position
        self.chromosome = chromosome
        self.start = int(start)
        self.end = int(end)
        self.REFERENCE_GENOME = REREFERENCE_GENOME

        # BAM file
        self.bam_path = bam_path

        # Encoded pileup placeholder
        self.pileup_encoded = np.zeros((IMAGE_HEIGHT, PLACEHOLDER_WIDTH)).astype(int)

        # Encoded qualities placeholder
        self.quality_encoded = np.zeros((IMAGE_HEIGHT, PLACEHOLDER_WIDTH)).astype(int)

        # Run encode
        self.encode_pileup()
        
    # Variant region
    @property
    def region_start(self):
        return self.start - OVERHANG - 2
    
    @property
    def region_end(self):
        return self.start + OVERHANG - 2

    @property
    def region_reference_sequence(self):
        return self.REFERENCE_GENOME.fetch(self.chromosome, self.region_start+1, self.region_end+2)

    @property
    def target_range(self):
        return range(self.start-1, self.end-1)

    # BAM file
    @property
    def bam_data(self):
        return pysam.AlignmentFile(self.bam_path, "rb")

    @property
    def start_coverage(self):
        start_coverage_arrays = self.bam_data.count_coverage(self.chromosome, self.start-1, self.start)
        return sum([coverage[0] for coverage in start_coverage_arrays])

    def decode_pileup(self):
        pileup_decoded = bcolors.OKGREEN + bcolors.UNDERLINE + self.region_reference_sequence + bcolors.ENDC
        pileup_decoded += '\n'

        for row in self.pileup_encoded:
            for base in row:
                base_decoded = decode_base(base)
                pileup_decoded += base_decoded
            pileup_decoded += '\n'
        return pileup_decoded

    def encode_pileup(self):
        for idx, read in enumerate(self.bam_data.fetch(reference=self.chromosome, start=self.start, end=self.end)):
            if idx >= IMAGE_HEIGHT:
                break
            self.pileup_encoded[idx, :], self.quality_encoded[idx, :] = (
                self._get_read_encoding(read, False)
            )
            
    def _get_read_encoding(self, read, debug=False):
        self._read = read
        # read properties
        self._cigar = self._read.cigar
        self._seq = self._read.seq
        self._query_qualities = np.array(self._read.query_qualities).astype(int)
        self._mapq = self._read.mapq

        # setting initial zeros for pileup and quality
        pileup = np.zeros((PLACEHOLDER_WIDTH, ))
        quality = np.zeros((PLACEHOLDER_WIDTH, ))

        # get reference genome for read
        self._ref = self.REFERENCE_GENOME.fetch(self.chromosome, self._read.reference_start, self._read.reference_start + 2*len(read.seq))
        
        # offset if reference_start before interest area [start - OVERHANG - 1, start + OVERHANG -1]
        offset = max(0, (self.start - OVERHANG - 1) - read.reference_start)

        #offset if reference_start inside interest area [start - OVERHANG - 1, start + OVERHANG -1]
        offset_picture = max(0, read.reference_start - (self.start - OVERHANG - 1))

        # pointers to reference genome positions
        genome_start_position = 0 + offset
        genome_end_position = 0

        # pointers to bases in read position
        base_start_position = self._calculate_base_start_position(genome_start_position, genome_end_position)
        base_end_position = 0

        # pointers to picture position
        picture_start_position = 0 + offset_picture
        picture_end_position = 0 + offset_picture
        
        # skip bad reads
        if not self._cigar or self._cigar[0][0] in (4, 5) or offset_picture > PLACEHOLDER_WIDTH:
            return pileup, quality

        #iterate over all cigar pairs
        for iter_num, (cigar_value, cigar_num) in enumerate(self._cigar):
            
            # update pointers end position
            base_end_position, genome_end_position = self._update_positions(
                cigar_value, cigar_num, base_end_position, genome_end_position
            )

            #we don't reach interest region
            if genome_end_position < genome_start_position:
                if base_start_position < base_end_position:
                    base_start_position = base_end_position
                continue
            
            # correction if we outside interest region
            genome_end_position = min(
                genome_end_position, 
                genome_start_position + PLACEHOLDER_WIDTH - picture_end_position
            )

            base_end_position = min(
                base_end_position, 
                base_start_position + PLACEHOLDER_WIDTH - picture_end_position
            )
            
            picture_step = min(
                cigar_num, 
                max(
                    genome_end_position - genome_start_position, 
                    base_end_position - base_start_position
                ))

            picture_end_position += picture_step

            # calculate quality
            quality[picture_start_position:picture_end_position] = self._calculate_quality(
                cigar_value, base_start_position, base_end_position, 
                genome_start_position, genome_end_position, picture_step
            )

            # calculate pilup
            pileup[picture_start_position:picture_end_position] = self._calculate_pileup(
                cigar_value, base_start_position, base_end_position, picture_step
            )

            # move pointers
            base_start_position = base_end_position
            genome_start_position = genome_end_position
            picture_start_position = picture_end_position

            if picture_end_position >= PLACEHOLDER_WIDTH:
                break

        return (pileup, quality)
    

    def _calculate_base_start_position(self, genome_start_position, genome_end_position):
        """
            Calculates base_start_position if genome_start_position > read.reference_start
        """
        if genome_start_position <= genome_end_position:
            return 0

        cigar_line = [cigar_value for cigar_value, cigar_num in self._cigar for x in range(cigar_num)]
        
        base_start_position = 0

        for cigar_value in cigar_line:
            base_start_position, genome_end_position = self._update_positions(
                cigar_value, 1, base_start_position, genome_end_position
            )

            if genome_end_position >= genome_start_position:
                break
        
        return base_start_position
    
    def _update_positions(self, cigar_value, cigar_num, base_position, genome_position):
        # match
        if cigar_value == 0:
            base_position += cigar_num
            genome_position += cigar_num
        # insertion
        elif cigar_value == 1:
            base_position += cigar_num
        # deletion
        elif cigar_value == 2:
            genome_position += cigar_num
        elif cigar_value == 4:
            base_position += cigar_num
            genome_position += cigar_num
        elif cigar_value == 5:
            base_position += cigar_num
        else:
            raise ValueError('Unsupported cigar value: {}'.format(cigar_value))

        return base_position, genome_position

    def _calculate_quality(self, cigar_value, 
                    base_start_position, base_end_position, 
                    genome_start_position, genome_end_position, picture_step):

        read = self._read
        start = self.start - 1 # WHY?
        end = self.end - 1 # WHY?
        query_qualities = self._query_qualities
        mapq = self._mapq
        ref = self._ref
        seq = self._seq
        
        absolute_genome_start = read.reference_start + genome_start_position
        absolute_genome_end = read.reference_start + genome_end_position
        current_genome_range = np.arange(absolute_genome_start, absolute_genome_end)

        current_quality = np.zeros_like((picture_step, ))

        # match
        if cigar_value == 0:
            current_quality = query_qualities[base_start_position:base_end_position] * mapq // 10
            matching_mask = (
                np.array(list(seq[base_start_position:base_end_position])) == 
                np.array(list(ref[genome_start_position:genome_end_position]))
            )
            non_interest_region = (current_genome_range < start) | (current_genome_range >= end)
            current_quality[matching_mask & non_interest_region] //= 3
        
        #insertion
        elif cigar_value == 1:
            current_quality = query_qualities[base_start_position:base_end_position] * mapq // 10

        #deletion
        elif cigar_value == 2:
            current_quality = np.ones((picture_step, ))*query_qualities[base_end_position] * mapq // 10
        
        return current_quality

    def _calculate_pileup(self, cigar_value, base_start_position, base_end_position, picture_step):
        current_pileup = np.zeros_like((picture_step, ))

        # match and insertion
        if cigar_value in (0, 1):
            #encode bases
            sub_seq = self._seq[base_start_position:base_end_position]
            current_pileup = self._get_encodings(cigar_value, sub_seq)
        
        #deletion 
        elif cigar_value == 2:
            #encode bases
            sub_seq = [-1] * picture_step
            current_pileup = self._get_encodings(cigar_value, sub_seq)
        
        return current_pileup

    def _get_encodings(self, cigar_value, bases):
        encoding_match = {
            'A': baseEncoder.A,
            'C': baseEncoder.C,
            'T': baseEncoder.T,
            'G': baseEncoder.G,
            'N': baseEncoder.EMPTY,
        }
        encoding_insertion = {
            'A': baseEncoder.IN_A,
            'C': baseEncoder.IN_C,
            'T': baseEncoder.IN_T,
            'G': baseEncoder.IN_G,
            'N': baseEncoder.IN_A,
        }

        result = np.zeros((len(bases), ))

        if cigar_value == 2:
            return result + baseEncoder.DEL

        for idx, base in enumerate(bases):
            if cigar_value == 0:
                result[idx] = encoding_match.get(base, 0)
            
            if cigar_value == 1:
                result[idx] = encoding_insertion.get(base, 0)
        
        return result
