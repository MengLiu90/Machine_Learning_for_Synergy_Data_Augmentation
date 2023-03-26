# DataCuration

## Pre-requisites
* Python3
* Pandas *(v1.1.3)*

## Step 1: Convert ENSP-ID in the chemical-protein links file to UniProt

* Download chemical-protein links (9606.protein_chemical.links.v5.0.1.tsv) file from STITCH dataset from http://stitch.embl.de
* seprate **STRING** and **Ensembl_PRO** from the total datasets

```
grep STRING 9606.protein_chemical.links.v5.0.1.tsv > string.data
grep Ensembl_PRO 9606.protein_chemical.links.v5.0.1.tsv > ensp.data
cat string.data ensp.data > idmapping.data
python protein_chemical.py
```
