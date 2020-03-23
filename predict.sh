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
  echo "    -g,--genome : Path to genome fasta file."
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
    -fb=*|--mother-bam=*)
    MOTHER_BAM="${i#*=}"
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


if [[ ${GENOME} = "" ]]; then
    echo "Error: GENOME --genome must be provided!"
    exit
fi

if [[ ${WORKDIR} = "" ]]; then
    echo "Error: WORKING DIRECTORY --workdir must be provided!"
    exit
fi

# Create intersected file
mkdir $WORKDIR

cp $CHILD_VCF $WORKDIR/child.vcf
bgzip $WORKDIR/child.vcf
tabix  -p vcf $WORKDIR/child.vcf.gz

cp $FATHER_VCF $WORKDIR/father.vcf
bgzip $WORKDIR/father.vcf
tabix  -p vcf $WORKDIR/father.vcf.gz

cp $MOTHER_VCF $WORKDIR/mother.vcf
bgzip $WORKDIR/mother.vcf
tabix  -p vcf $WORKDIR/mother.vcf.gz

bcftools isec -C $WORKDIR/child.vcf.gz $WORKDIR/father.vcf.gz $WORKDIR/mother.vcf.gz > $WORKDIR/intersected.txt

# Run P ython command
KERAS_BACKEND=tensorflow python ./main.py \
--mode=predict \
--genome=$GENOME \
--child-bam=$CHILD_BAM \
--father-bam=$FATHER_BAM \
--mother-bam=$MOTHER_BAM \
--intersected=$WORKDIR/intersected.txt \
--output=$WORKDIR/$OUTPUT

# Cleanup
rm $WORKDIR/child.vcf.gz*
rm $WORKDIR/father.vcf.gz*
rm $WORKDIR/mother.vcf.gz*
rm $WORKDIR/intersected.txt