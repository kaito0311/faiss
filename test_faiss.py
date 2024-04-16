import faiss 
import numpy as np 

np.random.seed(42) 

# index = faiss.IndexFlat(512, True)
# quantizer = faiss.IndexFlatL2(512)  # this remains the same
# index = faiss.IndexIVFPQ(quantizer, 512, 100, 8, 8)
index = faiss.index_factory(512, "IVF16,SQfp16")
index.set_include_quality()
index = faiss.IndexIDMap2(index)
xb = np.random.rand(1024, 512).astype(np.float32)
r_qua = np.random.rand(1024, 1).astype(np.float32) 


index.train(xb)
index.add_with_ids_with_quality(xb, r_qua, range(1024)) 
faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 16)

D, I = index.search_with_quality(xb[:10], 5, 0, 0.5)


for row in I: 
    for element in row: 
        print(element)
        print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")


D, I = index.search(xb[:10], 5)
print()

for row in I: 
    for element in row: 
        print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

print(I)


result_qua = index.reconstruct_qua(1)
print(result_qua)
print(r_qua[1])

# D, I = index.search(xb[:5], 10)
# print(index)
# print(I)
# print(D)
# print(index.boundary_search)
# D, I = index.boundary_search( x= xb[:5],k = 10,lower = 0.1, upper = 100.0, rm_duplicate = False, duplicate_thr = 1e-6)
# print(index)
# print(I)
# print(D)