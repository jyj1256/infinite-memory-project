import pinecone
import multiprocessing as mp
import numpy as np
import random
import string

from multiprocessing.pool import Pool, ThreadPool
from time import sleep

mp.set_start_method('fork')
print(mp.get_start_method())

pinecone.init()

INDEX_NAME = 'image-hybrid-search'
# INDEX_NAME = 'measure-512-rust'
# INDEX_NAME2 = 'measure-rest-1024'
index = pinecone.Index(INDEX_NAME)
# index2 = pinecone.Index(INDEX_NAME2)
print(index.describe_index_stats())
# print(index2.describe_index_stats())
# index.upsert([(None, np.random.random((768)).tolist(), None)])


# index.query(np.random.random(768).tolist(), top_k=5)

# rand_input = np.random.random((20, 768)).tolist()
# print(len(rand_input))
num_metadata_keys = 10

letters = string.ascii_letters + string.digits
metadata_keys = [''.join(random.choice(letters) for _ in range(10)) for _ in range(num_metadata_keys)]

def get_random_dict():
    def get_val(key):
        # Arbitrarily assumes that the first 3 metadata fields are textual, and the rest are floats
        if key in metadata_keys[:3]:
            # Generate random string
            return ''.join(random.choice(letters) for _ in range(random.randint(5, 20)))
        return f"{random.random() :.2f}"

    keys = random.sample(metadata_keys, random.randint(1, num_metadata_keys))
    return {k: get_val(k) for k in keys}

def generate_random_vectors(batch_size, vec_size):
    values = np.random.rand(batch_size, vec_size).tolist()
    ids = [''.join(random.choice(letters) for i in range(10)) for _ in range(batch_size)]
    metadata = [get_random_dict() for _ in range(batch_size)]
    vectors = list(zip(ids, values, metadata))
    return vectors

def do_query(vectors):
    # pinecone.init()
    # print(index.configuration.server_variables['index_name'])
    # print(pinecone.Config.API_KEY)
    # sleep(1)
    # res = index.query(vectors, top_k=5).to_dict()
    # print(res)
    # return res
    # return index.describe_index_stats()
    # print(id(index))
    my_index = pinecone.Index(INDEX_NAME)
    return my_index.query(vectors, top_k=5).to_dict()

def get_len(vectors):
    return len(vectors)

# with Pool(4) as mypool:
#     all_res = mypool.map(do_query, rand_input, chunksize=1)
#     # all_res = mypool.map(get_len, rand_input, chunksize=20)
#
# # print(index.describe_index_stats())
#
# print(type(all_res), len(all_res))
# # print(type(all_res[-1]))
# print(all_res)

# mypool = ThreadPool(4)
# mp_index = pinecone.Index(INDEX_NAME, pool_threads=mypool)
mp_index = pinecone.GRPCIndex(INDEX_NAME)
# rand_input = generate_random_vectors(50, 512)
# mp_index.upsert(rand_input)

mp_index.upsert([("mycoolval", [-120] * 768, {'key1': "I'm a string", 'key2': 3.0})])
# # mp_index2 = pinecone.GRPCIndex(INDEX_NAME2)
# print(mp_index.describe_index_stats())
# # print(mp_index2.describe_index_stats())

# rand_input = [('id', _, {}) for _ in rand_input]
# mp_index.upsert(rand_input)
# res = mp_index.query(np.random.random(768).tolist(), top_k = 5)
# print(res)

# res = mp_index.query((10 * np.ones(768)).tolist(), top_k = 5, include_metadata=False)
# print(res)

print(mp_index.query(id='id8', top_k=2))