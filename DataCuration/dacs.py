import pandas as pd
import os, sys
import time
import pickle
import multiprocessing as mp
import math

def mcc(old_prots, new_prots, nodes):
	TP = len(set([l for l in new_prots if l in old_prots]))
	FP = len(set([l for l in new_prots if l not in old_prots]))
	FN = len(set([l for l in old_prots if l not in new_prots]))
	TN = len(set([l for l in nodes if (l not in old_prots) and (l not in new_prots)]))
	MCC = float((TP*TN) - (FP*FN))/float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(0.5)
	return MCC


def process(file, target_dict, nodes):
	data = pd.read_csv(file)
	for cid in list(data.SIMILAR_CID.unique()):
		if cid in list(target_dict.keys()):
			old_prots = target_dict[file.replace(".smi.out.csv", "")]
			new_prots = target_dict[cid]
			try:
				MCC = mcc(old_prots, new_prots, nodes)
				data.loc[data.SIMILAR_CID==cid, "MCC"] = MCC
				if MCC > 0:
					DACS = math.sqrt((data[data.SIMILAR_CID==cid].TC.values[0])**2+(data[data.SIMILAR_CID==cid].MCC.values[0])**2)
					data.loc[data.SIMILAR_CID==cid, "DACS"] = DACS
				else:
					continue
			except:
				continue
		else:
			continue
	return data




if  __name__ == "__main__":
	start = time.time()
	
	with open("../MCC/targets_of_each_drug_in_stitch.pickle", "rb") as f:
		target_dict = pickle.load(f)
	
	ihp = pd.read_csv("../ihp_ppi.csv")
	nodes = list(set(list(ihp.Prot1.unique())+list(ihp.Prot2.unique())))
	
	file = "filename"
	data = process(file, target_dict, nodes)
	data.to_csv("dacs_"+file, index=False)
	print("Time: ", (time.time()-start))
