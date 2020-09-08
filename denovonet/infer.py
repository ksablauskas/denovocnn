from denovonet.settings import MINIMAL_COVERAGE
from denovonet.settings import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, MODEL_ARCHITECTURE, NUMBER_CLASSES

from denovonet.utils import get_variant_location
from denovonet.encoders import VariantClassValue, VariantInheritance
from denovonet.variants import SingleVariant, TrioVariant
from denovonet.models import get_model

from keras.models import load_model
from keras import backend as K

import time
import numpy as np
import pandas as pd
import os
import subprocess
from difflib import SequenceMatcher
import math

def get_variant_class(reference, alternate):
    if len(reference) == 1 and len(alternate) == 1:
        return VariantClassValue.snp
    elif len(reference) > len(alternate):
        return VariantClassValue.deletion
    elif len(reference) < len(alternate):
        return VariantClassValue.insertion
    else:
        if ',' in alternate:
            return VariantClassValue.deletion
        elif ',' in reference:
            return VariantClassValue.insertion
        else:
            return VariantClassValue.unknown

def get_end_coordinate(reference, start):
    return str( int(start) + len(reference) - 1 )

def parse_variant_from_intersected(row):
    chromosome = str(row[0])
    start = str(int(row[1]))
    reference = str(row[2])
    alternate = str(row[3])

    return chromosome, start, reference, alternate

def remove_matching_string(start, ref, var):
    match = SequenceMatcher(None, ref, var, autojunk=False).find_longest_match(0, len(ref), 0, len(var))
    
    insertion = False
    if len(ref) > len(var):
        insertion = True
        
    new_ref = ref.replace(ref[match.a: match.a + match.size], '', 1)
    new_var = var.replace(var[match.b:  match.b + match.size], '', 1)

    if insertion:
        start += match.size
    
    if match.size == 0:
        return start, new_ref, new_var
    else:
        return remove_matching_string(start, new_ref, new_var)


def split_comma_separated_variants(intersected_dataframe):
    original_intersected_array = np.array(intersected_dataframe)
    row_list = []

    for row in original_intersected_array:
        chromosome, start, ref, var, extra = row
        
        try:
            if ',' in var:
                for splitted_variant in var.split(','):
                    
                    new_start, new_ref, new_splitted_variant = remove_matching_string(start, ref, splitted_variant)          
                    new_row = chromosome, new_start, new_ref, new_splitted_variant, extra
                    row_list.append(new_row)
            else:
                new_start, new_ref, new_splitted_variant = remove_matching_string(start, ref, var)          
                new_row = chromosome, new_start, new_ref, new_splitted_variant, extra
                row_list.append(new_row)
        except TypeError as identifier:
            new_start, new_ref, new_splitted_variant = remove_matching_string(start, ref, var)          
            new_row = chromosome, new_start, new_ref, new_splitted_variant, extra
            row_list.append(new_row)
            pass
            
    splitted_intersected_array = np.array(row_list)

    return splitted_intersected_array

def infer_dnms_from_intersected(intersected_variants_tsv, child_bam, father_bam, mother_bam, REREFERENCE_GENOME, snp_model, in_model, del_model):

    print('SNP model',snp_model)
    print('Insertion model',in_model)
    print('Deletion model',del_model)
    
    # load models with this approach to avoid pytorch versions compatibility conflict 
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    
    model_snps = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
    model_insertions = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
    model_deletions = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
    
    model_snps.load_weights(snp_model)
    model_insertions.load_weights(in_model)
    model_deletions.load_weights(del_model)

    dnms_table = []
    start_time = time.time()

    intersected_dataframe = pd.read_csv(intersected_variants_tsv, sep='\t',names=['Chromosome','Position','Reference','Variant','extra'])
    intersected_dataframe = intersected_dataframe.replace(np.nan, '', regex=True)
    print(intersected_dataframe.head())
    
    splitted_intersected_array = split_comma_separated_variants(intersected_dataframe)

    dnm_counter = 0
    counter = 0
    for variant in splitted_intersected_array:
        counter += 1

        if counter % 100 == 0:
            elapsed = round((time.time() - start_time), 0)
            print('Variants evaluated: {} . Time elapsed: {}s'.format(counter, str(elapsed)))

        chromosome, start, reference, alternate = parse_variant_from_intersected(variant)
        variant_class = get_variant_class(reference, alternate)
        
        end = get_end_coordinate(reference, start)

        if variant_class == VariantClassValue.deletion or variant_class == VariantClassValue.insertion:
            if variant_class == VariantClassValue.insertion:
                end = str(int(end) + 1)

        
        loc = get_variant_location(chromosome, start, end)

        child_variant = SingleVariant(str(chromosome), int(start), int(end)+1, child_bam, REREFERENCE_GENOME)
        father_variant = SingleVariant(str(chromosome), int(start), int(end)+1, father_bam, REREFERENCE_GENOME)
        mother_variant = SingleVariant(str(chromosome), int(start), int(end)+1, mother_bam, REREFERENCE_GENOME)
        
        # Check variant start coverages
        if child_variant.start_coverage < MINIMAL_COVERAGE or father_variant.start_coverage < MINIMAL_COVERAGE or mother_variant.start_coverage < MINIMAL_COVERAGE:
            dnms_table_row = [chromosome, start, end, reference, alternate, -1]
            dnms_table.append(dnms_table_row)
            continue
        elif variant_class == VariantClassValue.unknown:
            dnms_table_row = [chromosome, start, end, reference, alternate, -2]
            dnms_table.append(dnms_table_row)
            continue
        else:

            mean_start_coverage = (child_variant.start_coverage + father_variant.start_coverage + mother_variant.start_coverage) / 3
            mean_start_coverage = int(round(mean_start_coverage))

            trio_variant = TrioVariant(child_variant, father_variant, mother_variant)

            if variant_class == VariantClassValue.snp:
                prediction = trio_variant.predict(model_snps)
            elif variant_class == VariantClassValue.deletion:
                prediction = trio_variant.predict(model_deletions)
            elif variant_class == VariantClassValue.insertion:
                prediction = trio_variant.predict(model_insertions)
            else:
                prediction_dnm = np.array([-2,-2])
            
            argmax = np.argmax(prediction, axis=1)
            
            if MODEL_ARCHITECTURE == 'advanced_cnn_binary':
                prediction_dnm = str(round(1.-prediction[0,0],3))
            else:
                prediction_dnm = str(round(prediction[0,0],3))

            dnms_table_row = [chromosome, start, end, reference, alternate, float(prediction_dnm), mean_start_coverage]
            dnms_table.append(dnms_table_row)

    K.clear_session()

    dnms_table = pd.DataFrame(dnms_table, columns=['Chromosome','Start position','End position','Reference','Variant','DNV probability','Mean Start Coverage'])
    return dnms_table
