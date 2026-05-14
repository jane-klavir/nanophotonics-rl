from numpy import load

data = load('data/processed/mie_dataset_v1.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])