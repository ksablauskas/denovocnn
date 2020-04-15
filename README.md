# DeNovoNet

A deep learning approach to calling de novo mutations (DNMs).

## Requirements

[Bcftools 1.8](https://samtools.github.io/bcftools/)
[bgzip 1.4.1](http://www.htslib.org/doc/bgzip.html)
[tabix 0.2.6](http://www.htslib.org/doc/tabix.html)

Python 3.5  
Tensorflow 1.10.0  
Keras 2.2.2  
Pysam  0.8.4  
Pandas 0.22.0  
Pillow 5.2.0  
Opencv 3.4.2 

## Installation
Recommended way of installing is creating an [Anaconda](https://www.anaconda.com/) environment.

```bash
#Create environment 
conda create -n denovocnn python=3.5

#Activate environment 
conda activate denovocnn

#Install bcftools and tabix+bgzip (single package) 
conda install -c bioconda bcftools
conda install -c bioconda tabix

#Install Python libraries
conda install -c conda-forge tensorflow=1.10.0
conda install -c conda-forge keras=2.2.2
conda install -c bioconda pysam
conda install -c anaconda pandas
conda install -c anaconda pillow
conda install -c conda-forge opencv
```

## Usage

### Training
You can train your own network by passing tab-separated files with the following columns:
<ul>
    <li><b>Chromosome</b> - variant chromsome.</li>
    <li><b>Start position</b> - variant start position.</li>
    <li><b>End position</b> - variant end position.</li>
    <li><b>End position</b> - variant end position.</li>
    <li><b>Reference</b> - reference allele.</li>
    <li><b>Variant</b> - variant allele.</li>
    <li><b>De novo assessment</b> - variant inheritance type. Use MV, PV, PV MV, PATERNAL, MATERNAL or SHARED for inherited variants and DNM for <i>de novo</i> mutations.</li>
    <li><b>Child</b> - path to child's BAM file for this variant.</li>
    <li><b>Father</b> - path to fagther's BAM file for this variant.</li>
    <li><b>Mother</b> - path to mother's BAM file for this variant.</li>
</ul>

```bash
KERAS_BACKEND=tensorflow python main.py \
--mode=train \
--build-dataset \
--genome=<PATH_TO_GENOME_FASTA_FILE> \
--train-dataset=<PATH_TO_TRAINING_DATASET_TSV> \
--val-dataset=<PATH_TO_VALIDATION_DATASET_TSV> \
--images=<PATH_TO_FOLDER_SAVING_IMAGES> \
--dataset-name=<DATASET_NAME>

```

### Prediction

```bash
./predict.sh \
-w <WORKING_DIRECTORY> \
-cv <CHILD_VCF> \
-fv <FATHER_VCF> \
-mv <MOTHER_VCF> \
-cb <CHILD_BAM> \
-fb <FATHER_BAM> \
-mb <MOTHER_BAM> \
-sm <SNP_MODEL> \
-im <INSERTION_MODEL> \
-dm <DELETION_MODEL> \
-g <REFERENCE_GENOME> \
-o <OUTPUT_FILE_NAME> \
```
OR
```bash
KERAS_BACKEND=tensorflow python main.py \
--mode=predict \
--genome=<PATH_TO_GENOME_FASTA_FILE> \
--child-bam=<PATH_TO_FATHER_BAM> \
--father-bam=<PATH_TO_FATHER_BAM> \
--mother-bam=<PATH_TO_MOTHER_BAM> \
--intersected=<PATH_TO_INTERSECTED_FILE> \
--mother-bam=<PATH_TO_MOTHER_BAM> \
--snp-model=<PATH_TO_SNP_MODEL> \
--in-model=<PATH_TO_INSERTION_MODEL> \
--del-model=<PATH_TO_DELETION_MODEL>
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)