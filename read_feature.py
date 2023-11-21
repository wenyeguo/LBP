import pickle
# load data from filename
def load_data(filename):
    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()
    # print("data read from file", data)
    return data

data = load_data("./features/substrings")
print("substrings length", len(data))
for i in range(10):
    print(data[i])