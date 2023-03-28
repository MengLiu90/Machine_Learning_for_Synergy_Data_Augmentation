import pandas as pd
import os, sys
import subprocess as s
import math
import multiprocessing as mp

def slice_data(data, nprocs):
	aver, res = divmod(len(data), nprocs)
	nums = []
	for proc in range(nprocs):
		if proc < res:
			nums.append(aver + 1)
		else:
			nums.append(aver)
	count = 0
	slices = []
	for proc in range(nprocs):
		slices.append(data[count: count+nums[proc]])
		count += nums[proc]
	return slices

def replace_both(aug1, aug2):
	cid1 = list(aug1.CID_1)
	cid2 = list(aug2.CID_2)
	pairs = [(x, y) for x in cid1 for y in cid2 if x != y]
	return pairs


def replace_single(synergy_inp, dacs):
	aug11 = pd.DataFrame()
	aug22 = pd.DataFrame()
	aug1212 = pd.DataFrame()
	for i, row in synergy_inp.iterrows():
		cell, cid1, cid2, synergy_score, tissue = row["Cell_Line"], row["CID_1"], row["CID_2"], row["SYNERGY_SCORE"], row["Tissue"]
		aug1 = dacs[dacs.ORIGINAL_CID==cid1][["SIMILAR_CID"]] #, "TC", "MCC", "DACS"]]
		aug1 = aug1.rename(columns={"SIMILAR_CID":"CID_1"})
		aug1.insert(1, "CID_2", cid2)
		aug1.insert(0, "Cell_Line", cell)
		aug1[["Original_pair", "SYNERGY_SCORE", "Tissue"]] = [cid1+"_"+cid2, synergy_score, tissue]
		aug2 = dacs[dacs.ORIGINAL_CID==cid2][["SIMILAR_CID"]] #, "TC", "MCC", "DACS"]]
		aug2 = aug2.rename(columns={"SIMILAR_CID":"CID_2"})
		aug2.insert(0, "CID_1", cid1)
		aug2.insert(0, "Cell_Line", cell)
		aug2[["Original_pair", "SYNERGY_SCORE", "Tissue"]] = [cid1+"_"+cid2, synergy_score, tissue]
		pairs = replace_both(aug1, aug2)
		aug12 = pd.DataFrame(pairs, columns=["CID_1", "CID_2"]) #columns = ["Cell_Line", "CID_1", "CID_2", "Original_pair", "SYNERGY_SCORE", "Tissue"])
		aug12[["Original_pair", "SYNERGY_SCORE", "Tissue"]] = [cid1+"_"+cid2, synergy_score, tissue]
		aug12.insert(0, "Cell_Line", cell)
		aug1 = aug1[aug1['CID_1'] != aug1['CID_2']]
		aug11 = pd.concat([aug11, aug1], ignore_index=True)
		aug2 = aug2[aug2['CID_1'] != aug2['CID_2']]
		aug22 = pd.concat([aug22, aug2], ignore_index=True)
		aug12 = aug12[aug12['CID_1'] != aug12['CID_2']]
		aug1212 = pd.concat([aug1212, aug12], ignore_index=True)
		'''
		if len(aug1) != 0:
			aug1 = aug1[aug1['CID_1'] != aug1['CID_2']]
			aug11 = pd.concat([aug11, aug1], ignore_index=True)
		elif len(aug2) != 0:
			aug2 = aug2[aug2['CID_1'] != aug2['CID_2']]
			aug22 = pd.concat([aug22, aug2], ignore_index=True)
		elif len(aug12) != 0:
			aug12 = aug12[aug12['CID_1'] != aug12['CID_2']]
			aug1212 = pd.concat([aug1212, aug12], ignore_index=True)
		else:
			continue
		'''
		#augmented_data = pd.concat([augmented_data, aug12], ignore_index=True)
		#augmented_data = augmented_data[augmented_data['CID_1'] != augmented_data['CID_2']]
	return aug11, aug22, aug1212

if __name__=="__main__":
	nprocs = mp.cpu_count()
	pool = mp.Pool(processes=nprocs)
	synergy = pd.read_csv("synergy_regression_tsu.csv")
	dacs = pd.read_csv("dacs_values_after_0.53_cutoff.csv")
	dacs = dacs[dacs.TC < 0.95]
	inp_lists = slice_data(synergy, nprocs)
	multi_result = [pool.apply_async(replace_single, (synergy_inp, dacs)) for synergy_inp in inp_lists]
	aug_out1 = pd.DataFrame()
	aug_out2 = pd.DataFrame()
	aug_out12 = pd.DataFrame()
	#augmented_data = pd.DataFrame()
	for p in multi_result:
		aug = p.get()
		aug1, aug2, aug12 = aug[0], aug[1], aug[2]
		aug_out1 = pd.concat([aug_out1, aug1], ignore_index=True)
		aug_out2 = pd.concat([aug_out2, aug2], ignore_index=True)
		aug_out12 = pd.concat([aug_out12, aug12], ignore_index=True)
	aug_out1.to_csv("augmented_cid1.csv", index=False)
	aug_out2.to_csv("augmented_cid2.csv", index=False)
	aug_out12.to_csv("augmented_cid12.csv", index=False)
