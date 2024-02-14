import yaml
import csv
import pandas as pd


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    file_path1 = 'configs/srcnn.yaml'
    file_path2 = 'configs/extranet.yaml'
    with open(file_path1, 'r') as file:
        data1 = yaml.safe_load(file)
        data1 = flatten_dict(data1)
        print(data1)
    df = pd.DataFrame.from_dict(data1, orient='index', columns=['A'])
    print(df)
    with open(file_path2, 'r') as file:
        data2 = yaml.safe_load(file)
        data2 = flatten_dict(data2)
        print(data2)
    df.insert(1, "B", data2)
    print(df)
    df.to_csv('results.csv')


if __name__ == '__main__':
    main()
