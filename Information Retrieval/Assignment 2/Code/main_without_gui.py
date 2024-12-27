import os
import pickle
from indexer import VectorSpaceModel


filepath = "/home/owaisk4/Win_backup/FAST NU assignments/Information Retrieval/Assignment 2/ResearchPapers"
saved_index = os.path.join(filepath, "vector_space_index.pkl")

if __name__ == "__main__":

    model: VectorSpaceModel
    if os.path.exists(saved_index):
        with open(saved_index, "rb") as f:
            model = pickle.load(f)
        print("Loaded vector space model from file")
    else:
        files = os.listdir(filepath)
        files = [os.path.join(filepath, file) for file in files]    
        model = VectorSpaceModel(files)
        print("Created vector space model from scratch")
        with open(saved_index, "wb") as f:
            pickle.dump(model, f)
    
    query = "machine learning"
    result = model.process_query(query)
    if len(result) == 0:
        print("NIL")
    else:
        print(result)