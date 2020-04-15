# import pandas as pd
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run denovoCNN on a trio.')
parser.add_argument('--mode', default='train', type=str, help='Mode that is used to run DeNovoNet. Possible modes:\ntrain\npredict')

# Train mode arguments
parser.add_argument('--build-dataset', dest='build_dataset', action='store_true', help='Build new dataset that will be used for training.')
parser.add_argument('--use-dataset', dest='build_dataset', action='store_false', help='Use existing dataset.')
parser.add_argument('--continue-training', dest='continue_training', action='store_true', help='Continue training model - used for transfer learning.')

parser.add_argument('--train-dataset', dest='train_dataset', default='in_train_pm.txt', type=str, help='Path to TSV file that is used to build training data')
parser.add_argument('--val-dataset', dest='val_dataset', default='in_val.txt', type=str, help='Path to TSV file that is used to build val data')
parser.add_argument('--images', default='images/', type=str, help='Path to folder that contains images for training ot will be used to save images')
parser.add_argument('--dataset-name', dest='dataset_name', default='insertions_pm', type=str, help='Name of the dataset.')
parser.add_argument('--epochs', default=25, type=int, help='Name of epochs for training.')

# Train mode or evaluate mode arguments
parser.add_argument('--genome', default='hs37d5.fa', type=str, help='Path to reference genome file.')

parser.add_argument('--model-path', dest='model_path', default='snp.h5', type=str, help='Path to model for evaluation mode or training mode with --continue-training.')
parser.add_argument('--output-model-path', dest='output_model_path', default='insertions_pm2.transfer.gray.center.h5', type=str, help='Path to model for evaluation mode or training mode with --continue-training.')

# Predict mode arguments
parser.add_argument('--child-bam', dest='child_bam' ,default=None, type=str, help='Path to child BAM.')
parser.add_argument('--father-bam', dest='father_bam', default=None, type=str, help='Path to father BAM.')
parser.add_argument('--mother-bam', dest='mother_bam', default=None, type=str, help='Path to mother BAM.')

parser.add_argument('--intersected', default='intersected.txt', type=str, help='Path to intersected file.')

# Prediction models
parser.add_argument('--snp-model', dest='snp_model', default=None, type=str, help='Path to SNP model.')
parser.add_argument('--in-model', dest='in_model', default=None, type=str, help='Path to insertion model.')
parser.add_argument('--del-model', dest='del_model', default=None, type=str, help='Path to deletion model.')

parser.add_argument('--output', default='output.txt', type=str, help='Path to output file.')
args = parser.parse_args()

# SET ARGUMENTS
EPOCHS = args.epochs
IMAGES_FOLDER = args.images
DATASET_NAME = args.dataset_name

# DeNovoNet
from denovonet.settings import AUGMENTATION, WORKING_DIRECTORY
from denovonet.dataset import Dataset
from denovonet import models
from denovonet.infer import infer_dnms_from_intersected

from keras.models import load_model

import pysam

if __name__ == "__main__":

    if args.mode == 'train':

        output_model_path = args.output_model_path
        REREFERENCE_GENOME = pysam.FastaFile(args.genome)

        if args.build_dataset:
             # Create dataset
            train_variants_path = args.train_dataset
            val_variants_path = args.val_dataset

            print('Building training dataset based on file {}'.format(train_variants_path))
            train_dataset = Dataset(train_variants_path, 'train', REREFERENCE_GENOME)
            
            print('Building validation dataset based on file {}'.format(train_variants_path))
            val_dataset = Dataset(val_variants_path, 'val', REREFERENCE_GENOME)
            
            train_dataset.save_images(IMAGES_FOLDER, DATASET_NAME)
            val_dataset.save_images(IMAGES_FOLDER, DATASET_NAME)
        
        # Train model
        if args.continue_training:
            print('Continuing training model {} .'.format(args.model_path))
            model = models.train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, output_model_path, continue_training=True, input_model_path=args.model_path)
        else:
            print('Training new model.')
            model = models.train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, output_model_path)

    elif args.mode == 'predict':
        REREFERENCE_GENOME = pysam.FastaFile(args.genome)

        child_bam = args.child_bam
        father_bam = args.father_bam
        mother_bam = args.mother_bam

        snp_model = args.snp_model
        in_model = args.in_model
        del_model = args.del_model

        path_to_tsv = args.intersected
        output_path = args.output

        dnms_table = infer_dnms_from_intersected(path_to_tsv, child_bam, father_bam, mother_bam, REREFERENCE_GENOME, snp_model, in_model, del_model)

        dnms_table.to_csv(output_path, sep='\t',index=False)

        print('Full analysis saved as {}.'.format(output_path))
        dnms_only_table = dnms_table.loc[dnms_table['DNV probability'] >= 0.5]
        dnms_only_table.to_csv(output_path + '.filtered.txt', sep='\t',index=False)
        print('Predicted DNVs (probability >= 0.5) saved as {}.'.format(output_path + '.filtered.txt'))

    # elif args.mode == 'evaluate':
    #     # Evaluate model
    #     model = load_model(args.model_path)
    #     models.evaluate(model, IMAGES_FOLDER, DATASET_NAME)

    else:
        print('Error. Unknown mode: {} . Please choose one of the following:\ntrain\npredict'.format(args.mode))