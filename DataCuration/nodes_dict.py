import pandas as pd
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


def association(inp, total_ass, nodes, d):
	for k in inp:
		prots = [p for p in list(total_ass[total_ass.chemical==k]["protein"].unique()) if p in nodes]
		d[k] = prots
	return d


def Convert(tup, di):
	for a, b in tup:
		di.setdefault(a, []).extend(b)
	return di

if __name__ == '__main__':
	import time
	start = time.time()
	total_ass = pd.read_csv("9606.protein_chemical_links_uniprot.csv")
	ihp = pd.read_csv("ihp_ppi.csv")
	nodes = list(set(list(ihp.Prot1.unique())+list(ihp.Prot2.unique())))
	with open("cids.lst") as f:
		cids = f.readlines()

	cids = [l.rstrip() for l in cids]
	d = dict((k, []) for k in cids)
	nprocs = mp.cpu_count()
	pool = mp.Pool(processes=nprocs)
	inp_lists = slice_data(cids, nprocs)
	multi_result = [pool.apply_async(association, (inp, total_ass, nodes, d)) for inp in inp_lists]
	result = [(k,v) for p in multi_result for k,v in p.get().items() if v]
	dictionary = {}
	dictionary = Convert(result, dictionary)
	import pickle
	with open("targets_of_each_drug_in_stitch.pickle", "wb") as f:
		pickle.dump(dictionary, f)
	print("Time: ", (time.time()-start))
	


