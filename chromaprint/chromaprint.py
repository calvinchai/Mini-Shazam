import acoustid
import numpy as np
import requests

def fingerprinting(inputs):
    """
    input: a list of path to audios
    
    """
    results = []
    for data in inputs:
        results.append(acoustid.fingerprint_file(data, force_fpcalc=True)[1])
    return results

def search_api(query, limit=5):
    # Construct the URL with the index, query, and limit
    url = f'http://10.0.0.6:6081/main/_search?query={query}&limit={limit}'

    # Make the GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON response
        return response.json()
    else:
        # Handle errors (e.g., print error message)
        print("Error:", response.status_code, response.text)
        return None


def lookup(query, k=10):
    """
    query: a list of fingerprint of the audios
    k: top k results

    """
    results = []
    for q in query:
        fp_array = np.frombuffer(q[:len(q) // 4*4], dtype=np.uint32)
        data = fp_array.tolist()
        data = ','.join(map(str, data))
        result = search_api(data, k)
        # get all ids
        result = [r['id'] for r in result['results']]

        results.append(result)
    return results

if __name__ == "__main__":
    # Test
    print(lookup(fingerprinting([r"E:\cs682\data\fma_small\038\038896.mp3"])))