import asyncio
import pandas as pd
import numpy as np
import pinecone

async def main():
    delegates = []
    pinecone.init()
    index = pinecone.GRPCIndexIndex("MyName")
    df = pd.read_csv("data.csv")
    vectors = df.values.tolist() # Or something similar
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        future = index.upsert(vectors[i:i + batch_size], async_req = True)
        delegates.append(future.delegate)

    # You can print the time here, this loop should actually return within seconds

    vector_counts = await asyncio.gather(*delegates)
    return np.array(vector_counts).sum()

asyncio.run(main())