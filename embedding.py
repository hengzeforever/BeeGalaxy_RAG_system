EMBEDDING_URL = "http://test.2brain.cn:9800/v1/emb"

import requests

def local_embedding(inputs):
    """Get embeddings from the embedding service"""
    headers = {"Content-Type": "application/json"}
    data = {"texts": inputs}
    result = requests.post(EMBEDDING_URL, headers=headers, json=data).json()
    return result['data']['text_vectors']

if __name__ == '__main__':
    inputs = ["Hello, world!"]
    output = local_embedding(inputs)[0]
    print(output)
    print("Dim: ", len(output))

