# import pandas as pd
import os
import argparse

env_path = '/path/to/env' 
sbatch_partition = 'some_partition'
path_to_main = './main.py'
cpus_per_task = '16'
mem_per_cpu = '16100M'

# Parse arguments
parser = argparse.ArgumentParser(description='Generate sbatch jobs for genomes.')
parser.add_argument('--workdir', dest='workdir', default=None, type=str, help='Path to folder with intersected_xx.txt files')
parser.add_argument('--output', dest='output', default='output.txt', type=str, help='Path to output job files.')

parser.add_argument('--snp-model', dest='snp_model', default=None, type=str, help='Path to SNP model.')
parser.add_argument('--in-model', dest='in_model', default=None, type=str, help='Path to insertion model.')
parser.add_argument('--del-model', dest='del_model', default=None, type=str, help='Path to deletion model.')

parser.add_argument('--child-bam', dest='child_bam' ,default=None, type=str, help='Path to child BAM.')
parser.add_argument('--father-bam', dest='father_bam', default=None, type=str, help='Path to father BAM.')
parser.add_argument('--mother-bam', dest='mother_bam', default=None, type=str, help='Path to mother BAM.')

parser.add_argument('--genome', default='hs37d5.fa', type=str, help='Path to reference genome file.')

args = parser.parse_args()

# SET ARGUMENTS

def get_job_query(workdir, child_bam, father_bam, mother_bam, genome, snp_model, in_model, del_model, intersected, output, slurm_logpath, slurm_errorpath):

    JOB_code = """
#!/bin/bash

#SBATCH -J DeNovoNet_PREDICT_{intersected}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --output={slurm_logpath}
#SBATCH --error={slurm_errorpath}
#SBATCH --partition={sbatch_partition}

# export PATH={env_path}

KERAS_BACKEND=tensorflow python {path_to_main} \
--mode=predict \
--genome={genome} \
--child-bam={child_bam} \
--father-bam={father_bam} \
--mother-bam={mother_bam} \
--snp-model={snp_model} \
--in-model={in_model} \
--del-model={del_model} \
--intersected={intersected} \
--output={output}
""".format(
    intersected=intersected, 
    cpus_per_task=cpus_per_task, 
    mem_per_cpu=mem_per_cpu,
    slurm_logpath=slurm_logpath,
    slurm_errorpath=slurm_errorpath,
    sbatch_partition=sbatch_partition,
    env_path=env_path,
    path_to_main=path_to_main,
    genome=genome, 
    child_bam=child_bam,
    father_bam=father_bam,
    mother_bam=mother_bam, 
    snp_model=snp_model,
    in_model=in_model,
    del_model=del_model,
    output=output
    )
    
    return JOB_code.strip()

def get_all_partition_files(workdir):
    files = list(os.listdir(workdir))

    has_partitions = sum([1 if "intersected_part" in filename else 0 for filename in files]) > 0

    if has_partitions:
        return [filename for filename in files if "intersected_part" in filename]
    else:
        return [filename for filename in files if "intersected.txt" in filename]

    return []

for intersected_file in get_all_partition_files(args.workdir):
    intersected_full_path = os.path.join(args.workdir, intersected_file)
    
    jobs_file_name = "job_" + intersected_file
    jobs_save_path = os.path.join(args.workdir, 'jobs', jobs_file_name)
    
    output = args.output + "_" + intersected_file
    
    slurm_logs_filename = 'report_DeNovoNet_PREDICT_' + intersected_file
    slurm_logpath = os.path.join(args.workdir, 'logs', slurm_logs_filename)
    
    slurm_error_logs_filename = 'error_DeNovoNet_PREDICT_' + intersected_file
    slurm_errorpath = os.path.join(args.workdir, 'logs', slurm_error_logs_filename)

    with open(jobs_save_path, 'w') as f:
        f.write(
            get_job_query(
                args.workdir, args.child_bam, args.father_bam, args.mother_bam, 
                args.genome, args.snp_model, args.in_model, args.del_model, 
                intersected_full_path, output, slurm_logpath, slurm_errorpath
            )
        )
