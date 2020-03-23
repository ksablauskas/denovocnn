import os

def get_variant_location(chromosome, start, end):
    if 'chr' not in chromosome:
        chromosome = 'chr' + chromosome
    return chromosome + ':' + start + '-' + end

def generate_model_report():
    report = ''
    report += ''