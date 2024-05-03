import os

import time
import faiss 
import numpy as np 
from typing import List

np.random.seed(42) 

d = 512
nb = 1000000
nq = 100
xb = np.random.rand(nb, d).astype(np.float32)
r_qua = np.random.rand(nb, 1).astype(np.float32)
ids = np.array(range(nb))

nlist = 16
k = 10




folder_save_onram_wo_qua = "./testing/saved/onram/wo_qua" 
folder_save_onram_w_qua = "./testing/saved/onram/w_qua" 

folder_save_ondisk_w_qua = "./testing/saved/ondisk/w_qua" 
folder_save_ondisk_wo_qua = "./testing/saved/ondisk/wo_qua" 



def merge_ondisk(
    trained_index: faiss.Index, shard_fnames: List[str], ivfdata_fname: str
) -> None:
    """Add the contents of the indexes stored in shard_fnames into the index
    trained_index. The on-disk data is stored in ivfdata_fname"""
    assert not isinstance(
        trained_index, faiss.IndexIVFPQR
    ), "IndexIVFPQR is not supported as an on disk index."
    # merge the images into an on-disk index
    # first load the inverted lists
    ivfs = []
    for fname in shard_fnames:
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        index = faiss.read_index(fname, faiss.IO_FLAG_MMAP)
        index_ivf = faiss.extract_index_ivf(index)
        ivfs.append(index_ivf.invlists)

        # avoid that the invlists get deallocated with the index
        index_ivf.own_invlists = False

    # construct the output index
    index = trained_index
    index_ivf = faiss.extract_index_ivf(index)

    assert index.ntotal == 0, "works only on empty index"

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(
        index_ivf.nlist, index_ivf.code_size, ivfdata_fname
    )

    # merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)

    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

    # now replace the inverted lists in the output index
    index.ntotal = index_ivf.ntotal = ntotal
    index_ivf.replace_invlists(invlists, True)
    invlists.this.disown()


def compare_search(index_wo_qua, index_w_q, qua_1 = False, qua_2 = True):
    if qua_1: 
        print("[INFO] Search with quality ")
        start = time.time()
        D1, I1, Q1 = index_wo_qua.search_with_quality(xb[:nq], k, 0, 1.0)
        print("[INFO] Done quality ", time.time() - start)
        
    else: 
        print("[INFO] Search normal")
        start = time.time()
        D1, I1 = index_wo_qua.search(xb[:nq], k)
        print("[INFO] Done normal ", time.time() - start)
    
    if qua_2: 
        print("[INFO] Search with quality ")
        start = time.time()
        D2, I2, Q2 = index_w_q.search_with_quality(xb[:nq], k, 0, 1.0)
        print("[INFO] Done quality ", time.time() - start)
    else: 
        print("[INFO] Search normal")
        start = time.time()
        D2, I2 = index_w_q.search(xb[:nq], k)
        print("[INFO] Done normal ", time.time() - start)
        

    assert np.sum(I1 - I2) == 0.0
    assert np.sum(D1 - D2) == 0.0

def test_index_flat():
    # Train 
    index_flat_wo_qua = faiss.IndexFlatL2(d) 

    index_flat_w_qua = faiss.IndexFlatL2(d)
    index_flat_w_qua.set_include_quality() 

    # Search 
    index_flat_w_qua.add_with_quality(xb, r_qua)
    index_flat_wo_qua.add(xb) 
    compare_search(index_flat_wo_qua, index_flat_w_qua) 

    # Save 
    name_save= "index_flat.bin"
    os.makedirs(folder_save_onram_wo_qua, exist_ok = True)
    os.makedirs(folder_save_onram_w_qua, exist_ok = True)
    faiss.write_index(index_flat_wo_qua, os.path.join(folder_save_onram_wo_qua, name_save))
    faiss.write_index(index_flat_w_qua, os.path.join(folder_save_onram_w_qua, name_save))

    backup_index_flat_wo_qua = index_flat_wo_qua 

    del index_flat_w_qua
    del index_flat_wo_qua 

    index_flat_w_qua = faiss.read_index(os.path.join(folder_save_onram_w_qua, name_save))
    index_flat_wo_qua = faiss.read_index(os.path.join(folder_save_onram_wo_qua, name_save))

    compare_search(backup_index_flat_wo_qua, index_flat_w_qua)
    compare_search(index_flat_wo_qua, index_flat_w_qua) 
    
def test_index_ivfflat(): 

    # Train 
    index_flat_wo_qua = faiss.index_factory(d, "IVF16,Flat")

    index_flat_w_qua = faiss.index_factory(d, "IVF16,Flat")
    index_flat_w_qua.set_include_quality() 

    # Search 
    if not index_flat_w_qua.is_trained: 
        index_flat_w_qua.train(xb) 
    if not index_flat_wo_qua.is_trained: 
        index_flat_wo_qua.train(xb) 
    
    index_flat_w_qua.add_with_ids_with_quality(xb, r_qua)
    index_flat_wo_qua.add(xb) 
    compare_search(index_flat_wo_qua, index_flat_w_qua) 

    # Save 
    name_save= "index_flat.bin"
    os.makedirs(folder_save_onram_wo_qua, exist_ok = True)
    os.makedirs(folder_save_onram_w_qua, exist_ok = True)
    faiss.write_index(index_flat_wo_qua, os.path.join(folder_save_onram_wo_qua, name_save))
    faiss.write_index(index_flat_w_qua, os.path.join(folder_save_onram_w_qua, name_save))

    backup_index_flat_wo_qua = index_flat_wo_qua 

    del index_flat_w_qua
    del index_flat_wo_qua 

    index_flat_w_qua = faiss.read_index(os.path.join(folder_save_onram_w_qua, name_save))
    index_flat_wo_qua = faiss.read_index(os.path.join(folder_save_onram_wo_qua, name_save))

    compare_search(backup_index_flat_wo_qua, index_flat_w_qua)
    compare_search(index_flat_wo_qua, index_flat_w_qua) 

def test_index_ivfflat_ondisk_w_qua(): 

    # Train 
    index_flat_w_qua = faiss.index_factory(d, "IVF16,Flat")
    index_flat_w_qua.set_include_quality()
    # index_flat_w_qua = faiss.IndexIDMap(index_flat_w_qua)
    index_flat_w_qua.train(xb) 
    
    index_ivf_part = faiss.extract_index_ivf(index_flat_w_qua) 
    clustering_index = faiss.IndexFlatL2(d)
    clustering_index.reset() 
    index_ivf_part.clustering_index = clustering_index

    if index_flat_w_qua.is_trained is False: 
        index_flat_w_qua.train(xb) 
    
    os.makedirs(folder_save_ondisk_w_qua, exist_ok = True)
    os.makedirs(folder_save_onram_w_qua, exist_ok = True)

    name = "index_ivfsq"
    faiss.write_index(index_flat_w_qua, os.path.join(folder_save_ondisk_w_qua, name + ".bin"))
    
    del index_flat_w_qua
    
    index_flat_w_qua = faiss.read_index(os.path.join(folder_save_ondisk_w_qua, name + ".bin"))
    index_flat_w_qua.add_with_ids_with_quality(xb, r_qua, ids)

    index_flat_w_qua.search_with_quality(xb[:nq], k, 0, 1.0)    

    faiss.write_index(index_flat_w_qua, os.path.join(folder_save_onram_w_qua, name + ".bin"))

    del index_flat_w_qua

    index_flat_w_qua_only_trained = faiss.read_index(os.path.join(folder_save_ondisk_w_qua, name + ".bin"))
    
    # from faiss.contrib.ondisk import merge_ondisk
    
    merge_ondisk(
        index_flat_w_qua_only_trained, 
        [os.path.join(folder_save_onram_w_qua, name + ".bin")],
        os.path.join(folder_save_ondisk_w_qua, name + ".ivfdata")
    )
    # exit()
    print("[INFO] Write on disk index: ")
    faiss.write_index(index_flat_w_qua_only_trained, os.path.join(folder_save_ondisk_w_qua, name + ".bin"))
    compare_search(
        faiss.read_index(os.path.join(folder_save_onram_w_qua, name + ".bin")),
        faiss.read_index(os.path.join(folder_save_ondisk_w_qua, name + ".bin")),
        True,
        True
    )

def test_index_idmap(): 

    # Train 
    index_flat_wo_qua = faiss.index_factory(d, "IDMap,SQfp16")

    index_flat_w_qua = faiss.index_factory(d, "IDMap,SQfp16")
    index_flat_w_qua.set_include_quality() 

    # Search 
    if not index_flat_w_qua.is_trained: 
        index_flat_w_qua.train(xb) 
    if not index_flat_wo_qua.is_trained: 
        index_flat_wo_qua.train(xb) 
    
    index_flat_w_qua.add_with_ids_with_quality(xb, r_qua, ids)
    index_flat_wo_qua.add_with_ids(xb, ids) 
    compare_search(index_flat_wo_qua, index_flat_w_qua) 

    # Save 
    name_save= "index_flat.bin"
    os.makedirs(folder_save_onram_wo_qua, exist_ok = True)
    os.makedirs(folder_save_onram_w_qua, exist_ok = True)
    faiss.write_index(index_flat_wo_qua, os.path.join(folder_save_onram_wo_qua, name_save))
    faiss.write_index(index_flat_w_qua, os.path.join(folder_save_onram_w_qua, name_save))

    backup_index_flat_wo_qua = index_flat_wo_qua 

    del index_flat_w_qua
    del index_flat_wo_qua 

    index_flat_w_qua = faiss.read_index(os.path.join(folder_save_onram_w_qua, name_save))
    index_flat_wo_qua = faiss.read_index(os.path.join(folder_save_onram_wo_qua, name_save))

    compare_search(backup_index_flat_wo_qua, index_flat_w_qua)
    compare_search(index_flat_wo_qua, index_flat_w_qua)

def test_reconstruct_quality():
    name_factory = "IVF16,Flat"

    index_flat_w_qua = faiss.index_factory(d, name_factory)
    index_flat_w_qua.set_include_quality()
    index_flat_w_qua.set_direct_map_type(faiss.DirectMap.Hashtable)

    print(name_factory)

    index_flat_w_qua.train(xb)
    index_flat_w_qua.add_with_ids_with_quality(xb, r_qua, ids) 

    D, I = index_flat_w_qua.search_with_quality(xb[:nq], k, 0, 1.0)
    
    # for row in I:
    #     for element in row:
    #         r_recons = index_flat_w_qua.reconstruct_qua(int(ids[element]))
    #         true_r_qua = r_qua[ids[element]]
    #         print(r_recons, true_r_qua)
    
    print(name_factory)

def save_ondisk_ivf(path_trained_save, path_onram_save, name_model, include_qua, architect):

    index = faiss.index_factory(d, architect) 
    if include_qua: 
        index.set_include_quality() 
    
    index.train(xb) 
    
    index_ivf_part = faiss.extract_index_ivf(index) 
    clustering_index = faiss.IndexFlatL2(d) 
    clustering_index.reset() 
    index_ivf_part.clustering_index = clustering_index

    if index.is_trained is False: 
        index.train(xb) 

    os.makedirs(path_trained_save, exist_ok = True)
    os.makedirs(path_onram_save, exist_ok = True)

    faiss.write_index(index, os.path.join(path_trained_save, name_model + ".bin"))
    
    del index
    
    index = faiss.read_index(os.path.join(path_trained_save, name_model + ".bin"))
    index.add_with_ids_with_quality(xb, r_qua, ids)

    if include_qua: 
        index.search_with_quality(xb[:nq], k, 0, 1.0)    
        index.search(xb[:nq], k)
    else: 
        index.search(xb[:nq], k)

    faiss.write_index(index, os.path.join(path_onram_save, name_model + ".bin"))

    index_only_trained = faiss.read_index(os.path.join(path_trained_save, name_model + ".bin"))

    merge_ondisk(
        index_only_trained, 
        [os.path.join(path_onram_save, name_model + ".bin")],
        os.path.join(path_trained_save, name_model + ".ivfdata")
    )

    faiss.write_index(index_only_trained, os.path.join(path_trained_save, name_model + ".bin"))
    
    return faiss.read_index(os.path.join(path_trained_save, name_model + ".bin"))


def test_veloc_search_ondisk():

    architect = "IVF16,SQfp16"

    index_w_qua =  save_ondisk_ivf(folder_save_ondisk_w_qua, folder_save_onram_w_qua, "ivf_sqfp16", True, architect)
    index_wo_qua =  save_ondisk_ivf(folder_save_ondisk_wo_qua, folder_save_onram_wo_qua, "ivf_sqfp16", True, architect)

    faiss.ParameterSpace().set_index_parameter(index_w_qua, "nprobe", 128)
    faiss.ParameterSpace().set_index_parameter(index_wo_qua, "nprobe", 128)

    compare_search(
        index_wo_qua,
        index_w_qua,
    )

def test_other(): 
    index = faiss.read_index(os.path.join(folder_save_ondisk_w_qua, "ivf_sqfp16" + ".bin"))
    
    D, I, Q = index.search_with_quality(xb[:nq], 5, 0, 0.5)

    print(I)
    print(Q)

def test_boundary_search_with_quality(): 

    index_flat_w_qua = faiss.index_factory(d, "IVF16,Flat")
    index_flat_w_qua.set_include_quality() 

    print("Train")
    index_flat_w_qua.train(xb)
    print("Add")
    index_flat_w_qua.add_with_quality(xb, r_qua)
    print("Seach")
    D1, I1, Q1 = index_flat_w_qua.boundary_search_with_quality(xb[:10], k, lower = 80.0, upper = 100.0, rm_duplicate = False, lower_quality = 0.5,  upper_quality = 1.0)
    D2, I2 = index_flat_w_qua.search(xb[:10], k)

    print(D1) 
    # print(I1)
    print(Q1)

    # print(np.sum(D1 - D2))
    # print(np.sum(I1 - I2))

    


if __name__ == "__main__":
    # test_reconstruct_quality()

    # print("[INFO] Testing index flat L2: ")
    # test_index_flat()

    # print("[INFO] Testing index IVFFlatL2: ")
    # test_index_ivfflat()

    # print("[INFO] Testing index IVFFSQ ondisk: ")
    # test_index_ivfflat_ondisk_w_qua()

    # print("[INFO] Testing index IDMap: ")
    # test_index_idmap()

    # print("[INFO] Test velo search on disk: ") 
    # test_veloc_search_ondisk() 

    # print("[INFO] Testing other")
    # test_other()

    print("[INFO] Testing boundary search with quality")
    test_boundary_search_with_quality() 

