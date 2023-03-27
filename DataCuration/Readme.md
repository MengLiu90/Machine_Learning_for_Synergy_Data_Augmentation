# DataCuration

## Pre-requisites
* Python3
* Pandas *(v1.1.3)*
* IHP-PING PPI *(https://github.com/gkm-software-dev/post-analysis-tools.git)*
* openbabel *(v2.3.1, https://sourceforge.net/projects/openbabel/)*

## Step 1: Convert ENSP-ID in the chemical-protein links file to UniProt

* Download chemical-protein links (9606.protein_chemical.links.v5.0.1.tsv) file from STITCH dataset from http://stitch.embl.de
* seprate **STRING** and **Ensembl_PRO** from the total datasets

```
grep STRING 9606.protein_chemical.links.v5.0.1.tsv > string.data
grep Ensembl_PRO 9606.protein_chemical.links.v5.0.1.tsv > ensp.data
cat string.data ensp.data > idmapping.data
python protein_chemical.py
```
## Step 2: Extract all the targets for CIDs in chemical-protein links if target in IHP-PING
This step requires users to have IHP-PING PPI in a csv file downloaded in the same folder as the `make_nodes_dict.py`
```
python make_nodes_dict.py
```
## Step 3: Similarity search of CIDs from AZ-DREAM challenge against STITCH
Get all SMILES files from AZ-DREAM synergy data

1. Tanimoto Similarity Search (TSS)
```
import pandas as pd
import os, sys
cids = [l for l in os.listdir() if "CIDs" in l]
for cid in cids:
  os.system("openbabel-2.3.1/bin/babel "+cid+" stitch.smi -ofpt > "+cid+".out")
 ```
 2. Converting TSS outputs to CSV

```
out_list = [l for l in os.listdir() if "CIDs" in l]
for out in out_list:
  data = pd.read_csv(out, sep=" ")
  index = [l[0].split(" ")[0].replace(">", "") for l in list(data.index)]
  data.index = index
  data.reset_index(inplace=True)
  data.columns = ["SIMILAR_CID", "TC"]
  data = data[data.SIMILAR_CID.str.contains("CIDs")]
  data.to_csv(out+".csv", index=False)
 ```
## Step 4: Finding drug action and chemical similarity score (DACS) for original drug in a pair
#### Requirements 
* multiprocessing, math, pickle python modules
* target dictionary generated at **step 2**
* IHP-PING PPI CSV file
* __Input__: CSV file generated at **step 3**

Replace "filename" with the CSV input file

```
python original_dac.py
```
