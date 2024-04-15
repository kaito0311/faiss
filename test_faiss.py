import faiss 
import numpy as np 

index = faiss.IndexFlat(512, True)
quantizer = faiss.IndexFlatL2(512)  # this remains the same
# index = faiss.IndexIVFPQ(quantizer, 512, 100, 8, 8)
index = faiss.index_factory(512, "IVF16,SQfp16")
xb = np.random.rand(1024, 512).astype(np.float32)
r_qua = np.random.rand(1024, 1).astype(np.float32) 


index.train(xb)
index.add(xb, r_qua) 

# D, I = index.search(xb[:5], 10)
# print(index)
# print(I)
# print(D)
# print(index.boundary_search)
# D, I = index.boundary_search( x= xb[:5],k = 10,lower = 0.1, upper = 100.0, rm_duplicate = False, duplicate_thr = 1e-6)
# print(index)
# print(I)
# print(D)