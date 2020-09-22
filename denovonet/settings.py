import os

# IMAGE PARAMETERS
OVERHANG = 20 #Number of nucleotides to each side of the center
NUCLEOTIDES = 4

IMAGE_CHANNELS = 3 #child, father, mother
IMAGE_WIDTH = 4 * (2 * OVERHANG + 1)
IMAGE_HEIGHT = 160 #Pileup height

CHANNEL_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
SINGLETON_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

PLACEHOLDER_WIDTH = 2 * OVERHANG + 1

IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

MINIMAL_COVERAGE = 7

AUGMENTATION = 0

INHERITED_VARIANT_CLASSES = ['PV','Paternal','MV','Maternal','PV MV']
# INHERITED_VARIANT_CLASSES = ['PV','MV','PV MV']

BINARY_CLASSIFICATION = False
if BINARY_CLASSIFICATION:
    VARIANT_CLASSES = ['DNM','IV'] #IV - inherited variant
    CLASSIFICATION_NAME = 'binary'
    
else:
    # VARIANT_CLASSES = ['DNM','PV','MV','PV MV']
    VARIANT_CLASSES = ['DNM','IV']
    CLASSIFICATION_NAME = 'multiclass'
NUMBER_CLASSES = len(VARIANT_CLASSES)

MODEL_ARCHITECTURE = 'advanced_cnn_binary' #advanced_cnn or cnn or resnet_v2

WORKING_DIRECTORY = 'workdir'

# Current production

# Current development
DATA_NAME = 'deletions_{w}_{h}_aug_{a}'.format(cn=CLASSIFICATION_NAME, h=IMAGE_HEIGHT, w=IMAGE_WIDTH,a=AUGMENTATION)

TRAINING_VALIDATION_SPLIT = 0.8
BATCH_SIZE = 32

TRAINING_X = os.path.join(WORKING_DIRECTORY,'data','{dn}_training_x.npy'.format(dn=DATA_NAME))
TRAINING_Y = os.path.join(WORKING_DIRECTORY,'data','{dn}_training_y.npy'.format(dn=DATA_NAME))
VALIDATION_X = os.path.join(WORKING_DIRECTORY,'data','{dn}_validation_x.npy'.format(dn=DATA_NAME))
VALIDATION_Y = os.path.join(WORKING_DIRECTORY,'data','{dn}_validation_y.npy'.format(dn=DATA_NAME))

MODEL_NAME = '{ma}_{cn}_{dn}.h5'.format(ma=MODEL_ARCHITECTURE, cn=CLASSIFICATION_NAME, dn=DATA_NAME)
DEVELOPMENNT_MODEL_PATH = os.path.join(WORKING_DIRECTORY,'models','development',MODEL_NAME)

# Production models

# Dev models
PRODUCTION_SNP_MODEL_PATH = os.path.join(WORKING_DIRECTORY,'models','development','snps.gray.filtered.notl.45.h5')
PRODUCTION_INSERTION_MODEL_PATH = os.path.join(WORKING_DIRECTORY,'models','development','insertions.gray.filtered.tl.15.h5')
PRODUCTION_DELETION_MODEL_PATH = os.path.join(WORKING_DIRECTORY,'models','development','deletions.gray.filtered.tl.15.h5')
