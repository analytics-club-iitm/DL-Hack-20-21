import pandas as pd
from typing import Dict, List
import json
import requests
import numpy as np
import sys
import os
from sklearn.decomposition import PCA

# DO NOT MODIFY
pca = PCA(n_components=32)
URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16

# split text into batches
def chunkify(test_csv, chunk_size=MAX_BATCH_SIZE):
    for i in range(0, len(test_csv), chunk_size):
        chunk = []
        for j in range(chunk_size):
            indx = i+j
            if indx == len(test_csv):
                break
            chunk.append(
                {
                    "paper_id": indx,
                    "title": test_csv["title"][indx],
                    "abstract": test_csv["abstract"][indx]
                }
            )
        yield chunk

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Incorrect arguments. Please run as: python3 submit_track2.py <path-to-predicted-csvfile>")
    
    # CREATING TEST DSET
    test_csv = sys.argv[1]
    if not os.path.exists(test_csv):
        raise ValueError("Prediction file does not exist")

    test_csv = pd.read_csv(test_csv)
    embeddings = []
    cntr = 0
    for chunk in chunkify(test_csv):
        response = requests.post(URL, json=chunk)
        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")
        for paper in response.json()["preds"]:
            embeddings.append(paper["embedding"])
            print(f"[{cntr}/{len(test_csv)}]", end="\r")
            cntr += 1
    print("Done... Creating submission file")

    embeddings = np.array(embeddings)
    embeddings = pca.fit_transform(embeddings)
    
    df = pd.DataFrame.from_records(embeddings)
    df["id"] = np.arange(len(embeddings))
    df.to_csv("submission.csv", index=False)
    print("Submission file created at ./submission.csv")