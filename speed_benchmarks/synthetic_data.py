
def sequence_data(num_docs=50, length=8000):
    return multi_label_sequence_data(num_docs=num_docs, length=length, num_labels=1)
    
def multi_label_sequence_data(num_docs=50, length=8000, num_labels=5):
    string = "The quick brown fox jumped over the lazy dog. "
    multiplier = length // len(string)
    x = [string * multiplier] * num_docs
    yi = []
    for i in range(multiplier):
        o = i * len(string)
        for lab_i in range(num_labels):
            yi += [
                {"start": 16 + o, "end": 19 + o, "label": str(lab_i), "text": "fox"},
                {"start": 41 + o, "end": 44 + o, "label": str(lab_i), "text": "dog"}
            ]
    y = [yi] * num_docs	
    return x, y

def classification_data(num_docs=50, length=1024):
    
    string1 = "The quick brown fox jumped over the lazy dog. "
    string1 = string1 * (length // len(string1))
    string2 = "The slow brown cat leaped over the active rabbit. "
    string2 = string1 * (length // len(string2))
    
    x = [string1, string2] * (num_docs // 2)
    y = ["lazy", "active"] * (num_docs // 2)

    return x, y
    

    
