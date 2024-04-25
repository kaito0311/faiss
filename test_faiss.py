import faiss 
import numpy as np 

np.random.seed(42) 

d = 512
nb = 1024
xb = np.random.rand(nb, d).astype(np.float32)
r_qua = np.random.rand(nb, 1).astype(np.float32)
ids = np.array(range(nb))
nlist = 16

def test_index_ivfsq():
    index = faiss.index_factory(512, "IVF16,SQfp16")
    index.set_include_quality()
    index = faiss.IndexIDMap2(index)

    index.train(xb)
    index.add_with_ids_with_quality(xb, r_qua, ids) 

    faiss.write_index(index, "./index_ivfsq.bin")

    faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 16)

    D1, I1 = index.search_with_quality(xb[:10], 5, 0, 1.0)

    for row in I1: 
        for element in row: 
            print(element)
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    D2, I2 = index.search(xb[:10], 5)
    print()

    for row in I2: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    print(np.sum(I1 - I2))

    del index

    index = faiss.read_index("./index_ivfsq.bin")
    faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 16)

    D3, I3 = index.search_with_quality(xb[:10], 5, 0, 1.0)
    for row in I3: 
        for element in row: 
            print(element)
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    D4, I4 = index.search(xb[:10], 5)
    print()

    for row in I4: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    print(np.sum(I1 - I3))
    print(np.sum(I2 - I4))


    from faiss.contrib.ondisk import merge_ondisk

    index_trained = faiss.read_index("./index_ivfsq.bin")
    merge_ondisk(
        index_trained, 
        []
    )

    pass 

def test_index_ivfsq_search_ondisk():
    index = faiss.index_factory(512, "IVF16,SQfp16")
    index.set_include_quality()
    index = faiss.IndexIDMap2(index)

    index.train(xb)

    faiss.write_index("./index_ivfsq_trained.bin")

    index.add_with_ids_with_quality(xb, r_qua, ids) 

    faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 16)

    D1, I1 = index.search_with_quality(xb[:10], 5, 0, 1.0)

    for row in I1: 
        for element in row: 
            print(element)
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    D2, I2 = index.search(xb[:10], 5)
    print()

    for row in I2: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    print(np.sum(I1 - I2))

    del index

    index = faiss.write_index("./index_ivfsq_onram.bin")
    faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 16)

    D3, I3 = index.search_with_quality(xb[:10], 5, 0, 1.0)
    for row in I3: 
        for element in row: 
            print(element)
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    D4, I4 = index.search(xb[:10], 5)
    print()

    for row in I4: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    print(np.sum(I1 - I3))
    print(np.sum(I2 - I4))


    from faiss.contrib.ondisk import merge_ondisk

    index_trained = faiss.read_index("./index_ivfsq.bin")
    merge_ondisk(
        index_trained, 
        []
    )

    pass 

def test_index_ivfflat():
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.set_include_quality()
    index = faiss.IndexIDMap2(index) 

    index.train(xb)
    index.add_with_ids_with_quality(xb, r_qua, ids) 
    faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 16)

    D1, I1 = index.search_with_quality(xb[:10], 5, 0, 0.5)


    for row in I1: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    D2, I2 = index.search(xb[:10], 5)
    print()

    for row in I2: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    print(np.sum(I1 - I2))


def test_index_flat(): 
    index = faiss.IndexFlatL2(d)
    index.set_include_quality()
    index = faiss.IndexIDMap2(index) 


    index.add_with_ids_with_quality(xb, r_qua, ids) 


    D1, I1 = index.search_with_quality(xb[:10], 5, 0, 0.5)


    for row in I1: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    D2, I2 = index.search(xb[:10], 5)
    print()

    for row in I2: 
        for element in row: 
            print(f"Id: {element} quality: {r_qua[element]}; quality_reconstruct: {index.reconstruct_qua(int(element))[0]}")

    print(np.sum(I1 - I2))



if __name__ == "__main__":
    test_index_ivfsq()