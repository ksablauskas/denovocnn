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
    
    def encode_base(self, base, cigar_value, pileup_coordinates):
        assert cigar_value in [0, 1, 2], 'Unsupported cigar value: {}'.format(cigar_value)

        if cigar_value == 0: #SNPs          
            if base == 'A':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.A
            elif base == 'C':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.C
            elif base == 'T':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.T
            elif base == 'G':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.G
            elif base == 'N':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.EMPTY
            else:
                raise TypeError('Unknown base type: {}'.format(base))

        elif cigar_value == 1: #Insertions
            if base == 'A':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.IN_A
            elif base == 'C':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.IN_C
            elif base == 'T':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.IN_T
            elif base == 'G':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.IN_G
            # Set up insertions for N..
            elif base == 'N':
                self.pileup_encoded[pileup_coordinates] = baseEncoder.IN_A
            else:
                raise TypeError('Unknown base type: {}'.format(base))

        elif cigar_value == 2: #Deletions
            self.pileup_encoded[pileup_coordinates] = baseEncoder.DEL
        else:
            raise ValueError('Unsupported cigar value: {}'.format(cigar_value))

    def encode_quality(self, genomic_position_counter, base, cigar_value, pileup_coordinates, base_counter, base_quality, mapq):
        assert cigar_value in [0, 1, 2], 'Unsupported cigar value: {}'.format(cigar_value)

        # Get reference base
        reference_base = self.region_reference_sequence[genomic_position_counter]
        
        visualize_quality = False
        
        genomic_position = self.region_start + genomic_position_counter + 1

        # Check if position in visual range
        if genomic_position in self.target_range:# and self.variant_class != VariantClassValue.deletion and cigar_value != 2:
            visualize_quality = True

        # Handle SNPs
        if cigar_value == 0:
            if base != reference_base:
                visualize_quality = True
        # Handle Insertions and Deletions
        elif cigar_value == 1 or cigar_value == 2:
            visualize_quality = True
        
        # Visualize central region
        if visualize_quality:
            self.quality_encoded[pileup_coordinates] = base_quality * mapq // 10
        # Visualize non central region
        elif visualize_quality == False and cigar_value in [0,1]:
            self.quality_encoded[pileup_coordinates] = (base_quality * mapq // 10) // 3

    def encode_pileup(self):
        read_counter = 0

        # Iterate through reads in BAM file
        for read in self.bam_data.fetch(reference=self.chromosome, start=self.start, end=self.end):    
            
            if read_counter >= IMAGE_HEIGHT:
                break

            # Get read information
            cigar = read.cigar
            seq = read.seq
            query_qualities = read.query_qualities
            mapq = read.mapq
            
            try:
                if cigar[0][0] == 4 or cigar[0][0] == 5:
                    read_counter += 1
                    continue
            except IndexError:
                continue
                
            genomic_position_counter = 0
            base_counter = 0
            base_position_index = 0
            quality_position_index = 0

            cigar_seq = []
            for cigar_pair in cigar:
                cigar_value, base_number = cigar_pair
                for cigar_base_index in range(base_number):
                    cigar_seq.append(cigar_value)

            # assert len(cigar_seq) == len(seq), 'len(cigar_seq) != len(seq): {} {} {} {} {}'.format(len(cigar_seq), len(seq), cigar, cigar_seq, seq)

            genomic_position = read.reference_start

            # Add empty spots
            if read.reference_start > self.region_start:
                base_position_index += read.reference_start - self.region_start - 1
                genomic_position_counter += read.reference_start - self.region_start - 1
                genomic_position += read.reference_start - self.region_start - 1
                quality_position_index += read.reference_start - self.region_start - 1
            
            for cigar_pair in cigar:
                
                cigar_value, base_number = cigar_pair
                assert cigar_value in [0, 1, 2, 4, 5], 'Unsupported cigar value: {}'.format(cigar_value)

                for cigar_base_index in range(base_number):

                    if self.region_start >= genomic_position:
                        if cigar_value != 2 and cigar_value != 5:
                            base_counter += 1

                        if cigar_value == 0 or cigar_value == 2 or cigar_value == 4:
                            genomic_position += 1

                        continue
                    
                    if self.region_end == read.reference_start + base_counter:
                        break
                    
                    pileup_coordinates = (read_counter, base_position_index)

                    if cigar_value in [0, 1, 2] and read_counter < IMAGE_HEIGHT and base_counter < len(seq) and genomic_position_counter < len(self.region_reference_sequence):
                        
                        base = seq[base_counter]
                        base_quality = query_qualities[base_counter]

                        assert cigar_value in [0, 1, 2], 'Unsupported cigar value: {}'.format(cigar_value)

                        if base_position_index < PLACEHOLDER_WIDTH:
                            self.encode_base(base, cigar_value, pileup_coordinates)
                            self.encode_quality(genomic_position_counter, base, cigar_value, pileup_coordinates, base_counter, base_quality, mapq)

                    if cigar_value == 0:
                        base_counter += 1
                        genomic_position += 1
                        genomic_position_counter += 1
                    elif cigar_value == 1:
                        base_counter += 1
                    elif cigar_value == 2:
                        genomic_position_counter += 1    
                        genomic_position += 1               
                    elif cigar_value == 4:
                        base_counter += 1
                        genomic_position += 1
                        genomic_position_counter += 1
                    elif cigar_value == 5:
                        base_counter += 1

                    base_position_index += 1

            read_counter += 1

    def decode_pileup(self):
        pileup_decoded = bcolors.OKGREEN + bcolors.UNDERLINE + self.region_reference_sequence + bcolors.ENDC
        pileup_decoded += '\n'

        for row in self.pileup_encoded:
            for base in row:
                base_decoded = decode_base(base)
                pileup_decoded += base_decoded
            pileup_decoded += '\n'
        return pileup_decoded

class TrioVariant():
    
    def __init__(self, child_variant, father_variant, mother_variant):
        self.child_variant = child_variant
        self.father_variant = father_variant
        self.mother_variant = mother_variant

        self.vstacked_pileup_encoded = np.vstack((self.child_variant.pileup_encoded, self.father_variant.pileup_encoded, self.mother_variant.pileup_encoded))
        self.vstacked_quality_encoded = np.vstack((self.child_variant.quality_encoded, self.father_variant.quality_encoded, self.mother_variant.quality_encoded))

        # Create encoded arrays with insertions for child, father and mother
        self.child_pileup_encoded_insertions = self.vstacked_pileup_encoded[:IMAGE_HEIGHT]
        self.father_pileup_encoded_insertions = self.vstacked_pileup_encoded[IMAGE_HEIGHT:2*IMAGE_HEIGHT]
        self.mother_pileup_encoded_insertions = self.vstacked_pileup_encoded[2*IMAGE_HEIGHT:]

        self.child_quality_encoded_insertions = self.vstacked_quality_encoded[:IMAGE_HEIGHT]
        self.father_quality_encoded_insertions = self.vstacked_quality_encoded[IMAGE_HEIGHT:2*IMAGE_HEIGHT]
        self.mother_quality_encoded_insertions = self.vstacked_quality_encoded[2*IMAGE_HEIGHT:]

        # Create singleton variant images
        self.child_variant_image = self.create_singleton_variant_image(self.child_pileup_encoded_insertions, self.child_quality_encoded_insertions)
        self.father_variant_image = self.create_singleton_variant_image(self.father_pileup_encoded_insertions, self.father_quality_encoded_insertions)
        self.mother_variant_image = self.create_singleton_variant_image(self.mother_pileup_encoded_insertions, self.mother_quality_encoded_insertions)

        # Combine singleton images
        self.image = self.create_trio_variant_image()

    def encode_insertions(self, vstacked_pileup_encoded, vstacked_quality_encoded):
        def correct_hstacked_array(hstacked_array, column_index):
            column = hstacked_array[:,column_index + 1]
            for row_index, base in enumerate(column):
                if base in [baseEncoder.IN_A,baseEncoder.IN_C,baseEncoder.IN_T,baseEncoder.IN_G]:
                    hstacked_array[row_index, column_index:-1] = hstacked_array[row_index, column_index+1:]
            
            corrected_hstacked_array = hstacked_array[:,:-1]
            return corrected_hstacked_array

        def insert_empty_column(vstacked_array, column_index):
            column = vstacked_array.T[column_index]

            left_array = vstacked_array[:,:column_index]
            middle_array = np.zeros((len(column),1))
            right_array = vstacked_array[:,column_index:]

            hstacked_array = np.hstack((left_array, middle_array, right_array))
            return hstacked_array

        def get_corrected_hstacked_array(vstacked_array, column_index):
            hstacked_array = insert_empty_column(vstacked_array, column_index)
            corrected_hstacked_array = correct_hstacked_array(hstacked_array, column_index)
            return corrected_hstacked_array

        # # Iterate over trio variant columns
        for column_index in range(len(vstacked_pileup_encoded.T)):
            column = vstacked_pileup_encoded.T[column_index]
            for row_index, base in enumerate(column):
                # Check if one of insertion values are in column
                if base in [baseEncoder.IN_A,baseEncoder.IN_C,baseEncoder.IN_T,baseEncoder.IN_G]:
                    # Split, hstack and correct pileup
                    corrected_hstacked_array_pileup = get_corrected_hstacked_array(vstacked_pileup_encoded, column_index)

                     # Split, hstack and correct quality                
                    corrected_hstacked_array_quality = get_corrected_hstacked_array(vstacked_quality_encoded, column_index)
                    
                    # Update arrays
                    vstacked_pileup_encoded = corrected_hstacked_array_pileup
                    vstacked_quality_encoded = corrected_hstacked_array_quality

                    # Go to next column
                    break
        
        return vstacked_pileup_encoded, vstacked_quality_encoded

    def create_singleton_variant_image(self, variant_pileup, variant_quality):
        variant_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

        for row_index, row in enumerate(variant_pileup):
            for column_index, value in enumerate(row):
                pileup_coordinates = (row_index, column_index)

                base = variant_pileup[pileup_coordinates]
                pixel_value = variant_quality[pileup_coordinates]

                if base == baseEncoder.A or base == baseEncoder.IN_A:
                    variant_image[row_index, column_index * 4 + 0] = pixel_value
                elif base == baseEncoder.C or base == baseEncoder.IN_C:
                    variant_image[row_index, column_index * 4 + 1] = pixel_value
                elif base == baseEncoder.T or base == baseEncoder.IN_T:
                    variant_image[row_index, column_index * 4 + 2] = pixel_value
                elif base == baseEncoder.G or base == baseEncoder.IN_G:
                    variant_image[row_index, column_index * 4 + 3] = pixel_value
                elif base == baseEncoder.DEL:
                    variant_image[row_index, column_index*4:column_index*4+4] = pixel_value

        return variant_image

    def normalize_image(self,image):
        image = image.astype(float)
        image /= 255
        
        return image

    def create_trio_variant_image(self):
        image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        image[:,:,0] = self.child_variant_image
        image[:,:,1] = self.father_variant_image
        image[:,:,2] = self.mother_variant_image

        return image

    def predict(self, model):
        expanded_image = np.expand_dims(self.image, axis=0)
        normalized_image = expanded_image.astype(float) / 255
        prediction = model.predict(normalized_image)
        return prediction

    @staticmethod
    def display_image(image):
        cv2.imwrite('', image) 

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)

        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image_path, image):
        cv2.imwrite(image_path, image) 

    @staticmethod
    def predict_image_path(image_path, model):
        image = Image.open(image_path)
        normalized_image = np.array(image).astype(float) / 255
        expanded_image = np.expand_dims(normalized_image, axis=0)
        prediction = model.predict(expanded_image)
        return prediction