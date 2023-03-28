import pandas as pd
import os, sys

#Script to convert ENSp IDs into UniProt Ids
#Pre-requisits:
    # Need to download ID mapping file from the UniProt
    # https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/idmapping/

# Download 9606.protein_chemical.links.v5.0.1.tsv from STITCH (chemical-protein links)

def drug_protein_association_for_virtual_data():
	drug_data = pd.read_csv("9606.protein_chemical.links.v5.0.1.tsv", sep="\t")

	idmap = pd.read_csv("ensp.prot", sep="\t", header=None)
	string = idmap[idmap[1]=="STRING"]
	ensp = idmap[idmap[1]=="Ensembl_PRO"]
	for prot in list(drug_data.protein.unique()):
		if prot in list(string[2].unique()):
			drug_data.loc[drug_data.protein==prot,"protein"] = list(string[string[2]==prot][0])[0]
		elif prot.split(".")[1] in list(ensp[2].unique()):
			prot1 = prot.split(".")[1]
			drug_data.loc[drug_data.protein==prot,"protein"] = list(ensp[ensp[2]==prot1][0])[0]
		else:
			continue
	return drug_data

if  __name__ == "__main__":
	drug_data = drug_protein_association_for_virtual_data()
	drug_data.replace(r'-\d+', '', regex=True, inplace = True)
	drug_data.to_csv("9606.protein_chemical_links_uniprot.csv",index=False)
