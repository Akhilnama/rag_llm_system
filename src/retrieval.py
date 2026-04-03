def retrieve(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query])
    
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    
    return results