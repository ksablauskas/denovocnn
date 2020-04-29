import numpy as np
import pandas as pd
import cv2
import os

from denovonet.settings import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
from denovonet.settings import TRAINING_VALIDATION_SPLIT, TRAINING_X, TRAINING_Y, VALIDATION_X, VALIDATION_Y

from denovonet.variants import SingleVariant, TrioVariant
from denovonet.encoders import VariantInheritance
# from denovonet.local_utils import get_rumc_bam_path

class Dataset():

    """
    Generates X, Y dataset based on a list of variants in a TSV file

    ...
    
    """
    
    def __init__(self, variants_path, train_val, REFERENCE_GENOME):
        """
        Attributes
        ----------
        variants_path : str
            path to file containing a list of DNMs
        """
        self.variants_path = variants_path
        self.train_val = train_val
        self.REFERENCE_GENOME = REFERENCE_GENOME

        # Load data
        self.variants_dataframe = pd.read_csv(self.variants_path, sep='\t')
        self.variants_dataframe = self.variants_dataframe[['Child','Father','Mother','Chromosome','Start position','End position','De novo assessment','Reference','Variant']]
        self.variants_array = np.array(self.variants_dataframe)
        # self.variants_array = self.variants_array[:30]

        # Get number of variants and split index
        self.number_variants = len(self.variants_array)

        self.x, self.y = self.populate(self.variants_dataframe)
        # self.number_cases = self.get_number_of_cases()
        # self.split_index = self.get_split_index()

        # # Split into training / validation cohorts
        # self.training_ids, self.validation_ids = self.get_training_validation_ids()

        # self.training_dataframe = self.variants_dataframe.loc[self.variants_dataframe['Child'].isin(self.training_ids)]
        # self.validation_dataframe = self.variants_dataframe.loc[self.variants_dataframe['Child'].isin(self.validation_ids)]

        # # Report
        # self.unique_train = len(self.get_unique_ids(self.training_dataframe))
        # self.unique_val = len(self.get_unique_ids(self.validation_dataframe))

        # # Populate
        # self.x_train, self.y_train = self.populate(self.training_dataframe)
        # self.x_val, self.y_val = self.populate(self.validation_dataframe)

    def get_unique_ids(self, dataframe):
        unique_ids = dataframe.Child.unique()
        return unique_ids

    def get_number_of_cases(self):
        unique_ids = self.get_unique_ids(self.variants_dataframe)
        unique_ids_array = np.array(unique_ids)
        number_cases = len(unique_ids_array)

        return number_cases

    def get_split_index(self):
        
        split_index = int(round((TRAINING_VALIDATION_SPLIT * self.number_cases),0))
        return split_index

    def get_training_validation_ids(self):
        index_cases_dataframe = self.variants_dataframe.Child.unique()
        index_cases_array = np.array(index_cases_dataframe)
        np.random.shuffle(index_cases_array)

        training_cases, validation_cases = index_cases_array[:self.split_index], index_cases_array[self.split_index:]

        return training_cases, validation_cases

    def get_placeholders(self, number_variants):
        x = np.zeros((number_variants, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        y = np.zeros((number_variants))

        return x, y

    def get_image(self, chromosome, start, end, child_bam, father_bam, mother_bam):
        child_variant = SingleVariant(chromosome, start, end, child_bam, self.REFERENCE_GENOME)
        father_variant = SingleVariant(chromosome, start, end,father_bam, self.REFERENCE_GENOME)
        mother_variant = SingleVariant(chromosome, start, end, mother_bam, self.REFERENCE_GENOME)

        trio_variant = TrioVariant(child_variant, father_variant, mother_variant)

        return trio_variant.image

    def get_label(self, assessment):
        if  assessment in ['MATERNAL','MATERNAL - LOW COVERAGE','MATERNAL - NO EXACT MATCH','MV', 'PATERNAL','PATERNAL - LOW COVERAGE', 'PV', 'PV MV','SHARED','SHARED - LOW COVERAGE']:
            label = VariantInheritance.IV
        elif assessment == 'DNM':
            label = VariantInheritance.DNM
        else:
            raise('Unknown variant assessment {} {} {} {}'.format(assessment))

        return label

    def populate(self, cases_dataframe):
        # Iterate through variants and add them to placeholder
        
        case_array = np.array(cases_dataframe)
        number_variants = len(cases_dataframe)
        x, y = self.get_placeholders(number_variants)

        for index, row in enumerate(case_array):

            # First three columns for family IDs
            # TODO: lookup indexes for these values based on header
            child_id, father_id, mother_id = row[0], row[1], row[2]
            chromosome, start, end = row[3], row[4], row[5]
            assessment, reference_allele, variant_allele = row[6], row[7], row[8]

            # Adjust end coordinates for pysam
            end += 1

            # Get bam paths
            child_bam, father_bam, mother_bam = get_rumc_bam_path(child_id), get_rumc_bam_path(father_id), get_rumc_bam_path(mother_id)

            # Get image and labels
            image = self.get_image(chromosome, start, end, child_bam, father_bam, mother_bam)
            label = self.get_label(assessment)

            # Populate entry
            x[index] = image
            y[index] = label

            print('Loading variant {} : {} , value {}. {}:{}-{} {} {}'.format(index, assessment, label, chromosome, start, end, reference_allele, variant_allele))
        
        shuffled_x, shuffled_y = self.shuffle(x, y)

        return shuffled_x, shuffled_y

    @staticmethod
    def shuffle(x, y):
        
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        shuffled_x = x[randomize]
        shuffled_y = y[randomize]

        return shuffled_x, shuffled_y
    
    def split(self):
        
        shuffled_x, shuffled_y = self.shuffle()

        x_train = shuffled_x[:self.split_index]
        y_train = shuffled_y[:self.split_index]

        x_val = shuffled_x[self.split_index:]
        y_val = shuffled_y[self.split_index:]

        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val

    def save(self):

        np.save(TRAINING_X, self.x_train) 
        np.save(TRAINING_Y, self.y_train)

        np.save(VALIDATION_X, self.x_val) 
        np.save(VALIDATION_Y, self.y_val)

        print('Saved TRAINING X as {}. Number of variants {}'.format(TRAINING_X, len(self.x_train)))
        print('Saved TRAINING Y as {}. Number of variants {}'.format(TRAINING_Y, len(self.y_train)))

        print('Saved VALIDATION X as {}. Number of variants {}'.format(VALIDATION_X, len(self.x_val)))
        print('Saved VALIDATION Y as {}. Number of variants {}'.format(VALIDATION_Y, len(self.x_val)))

    def save_images(self, IMAGES_FOLDER, DATASET_NAME):
        train_val = self.train_val
        x_data = self.x
        y_data = self.y

        for index, image in enumerate(x_data):
            label = y_data[index]
            if label == VariantInheritance.IV:
                save_path = os.path.join(IMAGES_FOLDER, DATASET_NAME, train_val, 'iv')
            elif label == VariantInheritance.DNM:
                save_path = os.path.join(IMAGES_FOLDER, DATASET_NAME, train_val, 'dnm')
            else:
                raise('Unknown label {} {} {} {}'.format(label))
            
            filename = '{}.png'.format(str(index))
            
            # Save as RGB 
            reordered_placeholder = np.zeros((image.shape))
            reordered_placeholder[:,:,0], reordered_placeholder[:,:,1], reordered_placeholder[:,:,2] = image[:,:,2], image[:,:,1], image[:,:,0]

            output_path = os.path.join(save_path,filename)

            cv2.imwrite(output_path,reordered_placeholder)
            print('Saved {} as {}'.format(filename,output_path))
            
            
class CustomAugmentation(object):
    """ Defines a custom augmentation class. Randomly applies one of transformations."""
        
    def __init__(self, probability=0.9, reads_cropping = False, reads_shuffling = False, seed=None):
        self.probability = probability
        self.reads_cropping = reads_cropping
        self.reads_shuffling = reads_shuffling
        self.transformations = []
        
        if seed:
            np.random.seed(seed)

    def _check_augmentations(self):
        self.transformations = []

        if self.reads_cropping:
            self.transformations.append(self._reads_cropping)
        if self.reads_shuffling:
            self.transformations.append(self._reads_shuffling)

    @staticmethod
    def _reads_cropping(img):
        new_img = img.copy()

        nreads_c, nreads_f, nreads_m = tuple(np.sum(np.sum(new_img, axis=1) > 0., axis=0))

        n_reads_c = max(5, nreads_c) 
        n_reads_f = max(5, nreads_f) 
        n_reads_m = max(5, nreads_m)

        nreads_c = np.random.choice(np.arange(5, nreads_c + 1))
        nreads_f = np.random.choice(np.arange(5, nreads_f + 1))
        nreads_m = np.random.choice(np.arange(5, nreads_m + 1))

        new_img[nreads_c:, :, 0] = 0.
        new_img[nreads_f:, :, 1] = 0.
        new_img[nreads_m:, :, 2] = 0.

        return new_img

    @staticmethod
    def _reads_shuffling(img):
        new_img = img.copy()

        nreads_c, nreads_f, nreads_m = tuple(np.sum(np.sum(new_img, axis=1) > 0., axis=0))
        print (nreads_c, nreads_f, nreads_m)

        np.random.shuffle(new_img[:nreads_c, :, 0])
        np.random.shuffle(new_img[:nreads_f, :, 1])
        np.random.shuffle(new_img[:nreads_m, :, 2])
        
        return new_img

    def __call__(self, img):

        if img.shape[2] != 3:
            print (img.shape)
            raise Exception("Wrong image format!")
        
        random_number = np.random.random()
                
        if random_number > self.probability:
            pass
        else:
            self._check_augmentations()
            transformation = np.random.choice(self.transformations)

            return transformation(img)
                
        return img
