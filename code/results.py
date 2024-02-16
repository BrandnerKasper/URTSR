import yaml
import pandas as pd
import os
import glob


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


def create_csv(folder_path: str) -> None:
    d_list = []
    # Iterate over all YAML files in the folder
    for file_path in sorted(glob.glob(os.path.join(folder_path, '*.yaml'))):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            data = flatten_dict(data)
            # Process data as needed
            print(f"Processing {file_path}:\n{data}")
            d_list.append(data)

    # Create new dataframe from yaml
    # df = pd.DataFrame.from_dict(data, orient='index', columns=['1'])
    df = pd.DataFrame(d_list).T
    # Save to csv file
    df.to_csv("results/results.csv")


def main():
    folder_path = "results"

    create_csv(folder_path)


if __name__ == '__main__':
    main()
