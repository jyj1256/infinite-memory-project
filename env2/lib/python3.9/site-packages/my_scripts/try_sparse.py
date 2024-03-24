import pinecone

INDEX_NAME = 'amnon-sparse-test'

pinecone.init(api_key="724f029e-db08-427e-b503-c3e8088bf13c", environment="internal-beta")
print(pinecone.list_indexes())
print(pinecone.describe_index(INDEX_NAME))

index = pinecone.Index(INDEX_NAME)
data = [{'id': 'id1', 'values': [1.0, 2.0, 3.0]},
        {'id': 'id2', 'values': [11.0, 12.0, 1.3], 'metadata': {'key1': ['list', 'of', 'strings'], 'key2': 3, 'key3': 'bla bla'}},
        {'id': 'id3', 'values': [2.1, 2.2, 2.3], 'metadata': {'key1': ['again'], 'key2': 2, 'key3': 'bli bli'},
         'sparse_values': {'indices': [1, 2], 'values': [0.5, 0.2]}},
        {'id': 'id4', 'values': [3.1, 3.2, 3.3], 'sparse_values': {'indices': [1, 8], 'values': [0.5, 0.9]}},
        ]

print(index.upsert(data))

index = pinecone.GRPCIndex(INDEX_NAME)
print(index.upsert(data))
