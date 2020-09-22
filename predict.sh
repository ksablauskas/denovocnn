#!/usr/bin/env bash

# Help command
if [[ ("$1" == "-h") || ("$1" == "--help") ]]; then
  echo "Usage: ./`basename $0` [-wd] [-cv] [-fv] [-mv] [-cb] [-fb] [-mb] [-g] [-o]"
  echo "    -w,--workdir : Path to working directory."
  echo "    -cv,--child-vcf : Path to child vcf file."
  echo "    -fv,--father-vcf : Path to father vcf file."
  echo "    -mv,--mother-vcf : Path to mother vcf file."
  echo "    -cb,--child-bam : Path to child bam file."
  echo "    -fb,--father-bam : Path to father bam file."
  echo "    -mb,--mother-bam : Path to mother bam file."
  echo "    -sm,--snp-model : Path to SNP model."
  echo "    -im,--in-model : Path to insertion model."
  echo "    -dm,--del-model : Path to deletion model."
  echo "    -g,--genome : Path to genome fasta file."
  echo "    -m,--mode : Mode of running: 'genome' or 'exome'."
  echo "    -o,--output : Output file name (will be saved to workdir)."
  exit 0
fi

# Parse arguments
for i in "$@"
do
case $i in
    -w=*|--workdir=*)
    WORKDIR="${i#*=}"
    shift
    ;;
    -cv=*|--child-vcf=*)
    CHILD_VCF="${i#*=}"
    shift
    ;;
    -fv=*|--father-vcf=*)
    FATHER_VCF="${i#*=}"
    shift
    ;;
    -mv=*|--mother-vcf=*)
    MOTHER_VCF="${i#*=}"
    shift
    ;;
    -cb=*|--child-bam=*)
    CHILD_BAM="${i#*=}"
    shift
    ;;
    -fb=*|--father-bam=*)
    FATHER_BAM="${i#*=}"
    shift
    ;;
    -mb=*|--mother-bam=*)
    MOTHER_BAM="${i#*=}"
    shift
    ;;
    -sm=*|--snp-model=*)
    SNP_MODEL="${i#*=}"
    shift
    ;;
    -im=*|--in-model=*)
    IN_MODEL="${i#*=}"
    shift
    ;;
    -dm=*|--del-model=*)
    DEL_MODEL="${i#*=}"
    shift
    ;;
    -g=*|--genome=*)
    GENOME="${i#*=}"
    shift
    ;;
    -o=*|--output=*)
    OUTPUT="${i#*=}"
    shift
    ;;
    -m=*|--mode=*)
    MODE="${i#*=}"
    shift
    ;;
    --default) # is this needed?
    DEFAULT=YES
    shift
    ;;
    *)
          # unknown option
    ;;
esac
done

# Evaluate arguments
# VCFs
if [[ ${CHILD_VCF} = "" ]]; then
    echo "Error: Path to child vcf file --child-vcf must be provided!"
    exit
fi
if [[ ${FATHER_VCF} = "" ]]; then
    echo "Error: Path to father vcf file --father-vcf must be provided!"
    exit
fi
if [[ ${MOTHER_VCF} = "" ]]; then
    echo "Error: Path to mother vcf file --mother-vcf must be provided!"
    exit
fi

# BAMs
if [[ ${CHILD_BAM} = "" ]]; then
    echo "Error: Path to child bam file --child-bam must be provided!"
    exit
fi
if [[ ${FATHER_BAM} = "" ]]; then
    echo "Error: Path to father bam file --father-bam must be provided!"
    exit
fi
if [[ ${MOTHER_BAM} = "" ]]; then
    echo "Error: Path to mother bam file --mother-bam must be provided!"
    exit
fi

# MODELS
if [[ ${SNP_MODEL} = "" ]]; then
    echo "Error: Path to SNP model --snp-model must be provided!"
    exit
fi
if [[ ${IN_MODEL} = "" ]]; then
    echo "Error: Path to insertion model --in-model must be provided!"
    exit
fi
if [[ ${DEL_MODEL} = "" ]]; then
    echo "Error: Path to deletion model --del-model must be provided!"
    exit
fi


if [[ ${GENOME} = "" ]]; then
    echo "Error: GENOME --genome must be provided!"
    exit
fi

if [[ ${WORKDIR} = "" ]]; then
    echo "Error: WORKING DIRECTORY --workdir must be provided!"
    exit
fi

if [[ ${MODE} = "" ]]; then
    echo "Error: MODE --mode must be provided!"
    exit
fi

echo "Start preprocessing step"

# Create intersected file
mkdir $WORKDIR

if [ $CHILD_VCF =  *"gz" ]; then
    cp $CHILD_VCF $WORKDIR/child.vcf.gz
else
    bcftools sort $CHILD_VCF > $WORKDIR/child.vcf
    bgzip $WORKDIR/child.vcf
fi
BGZIPPED_CHILD_VCF=$WORKDIR/child.vcf.gz
tabix  -p vcf $BGZIPPED_CHILD_VCF

if [ $FATHER_VCF =  *"gz" ]; then
    cp $FATHER_VCF $WORKDIR/father.vcf.gz
else
    bcftools sort $FATHER_VCF > $WORKDIR/father.vcf
    bgzip $WORKDIR/father.vcf
fi
BGZIPPED_FATHER_VCF=$WORKDIR/father.vcf.gz
tabix  -p vcf $BGZIPPED_FATHER_VCF

if [ $MOTHER_VCF =  *"gz" ]; then
    cp $MOTHER_VCF $WORKDIR/mother.vcf.gz
else
    bcftools sort $MOTHER_VCF > $WORKDIR/mother.vcf
    bgzip $WORKDIR/mother.vcf
fi
BGZIPPED_MOTHER_VCF=$WORKDIR/mother.vcf.gz
tabix  -p vcf $BGZIPPED_MOTHER_VCF

bcftools isec -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF > $WORKDIR/intersected.txt

echo "Preprocessing step finished"

if [[ ${MODE} = "exome" ]]; then
    echo "Start DenovoCNN"

    # Run Python command
    KERAS_BACKEND=tensorflow python ./main.py \
    --mode=predict \
    --genome=$GENOME \
    --child-bam=$CHILD_BAM \
    --father-bam=$FATHER_BAM \
    --mother-bam=$MOTHER_BAM \
    --snp-model=$SNP_MODEL \
    --in-model=$IN_MODEL \
    --del-model=$DEL_MODEL \
    --intersected=$WORKDIR/intersected.txt \
    --output=$OUTPUT
    
    echo "DenovoCNN finished, output in:"
    echo $OUTPUT
else
    split -d -l 10000 --additional-suffix=.txt $WORKDIR/intersected.txt $WORKDIR/intersected_part
    mkdir $WORKDIR/logs
    mkdir $WORKDIR/jobs
    
    # Run Python command
    python ./genomes_job_generator.py \
    --workdir=$WORKDIR \
    --genome=$GENOME \
    --child-bam=$CHILD_BAM \
    --father-bam=$FATHER_BAM \
    --mother-bam=$MOTHER_BAM \
    --snp-model=$SNP_MODEL \
    --in-model=$IN_MODEL \
    --del-model=$DEL_MODEL \
    --output=$OUTPUT
    
    echo "Generated jobs for slurm in:"
    echo $WORKDIR/jobs
fi

# Cleanup
rm $WORKDIR/child.vcf.gz*
rm $WORKDIR/father.vcf.gz*
rm $WORKDIR/mother.vcf.gz*
