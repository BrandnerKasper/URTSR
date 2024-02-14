import yaml
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


def create_csv(data_file_path: str, csv_file_path: str) -> None:
    # Read from yaml file
    with open(data_file_path, 'r') as file:
        data = yaml.safe_load(file)
        data = flatten_dict(data)
    # Create new dataframe from yaml
    df = pd.DataFrame.from_dict(data, orient='index', columns=['1'])
    # Save to csv file
    df.to_csv(csv_file_path)


# TODO
def add_to_csv(data_file_path: str, csv_file_path: str) -> None:
    # Read from yaml file
    with open(data_file_path, 'r') as file:
        data = yaml.safe_load(file)
        data = flatten_dict(data)
    # Create dataframe from csv file
    df = pd.read_csv(csv_file_path)
    idx = df.shape[1]
    # Insert the config file data into the old csv
    df.insert(idx, idx+1, data)
    df.to_csv(csv_file_path)


def main():
    data_file_path1 = 'configs/srcnn.yaml'
    data_file_path2 = 'configs/extranet.yaml'
    csv_file_path = "results2.csv"

    create_csv(data_file_path1, csv_file_path)
    add_to_csv(data_file_path2, csv_file_path)


if __name__ == '__main__':
    main()
