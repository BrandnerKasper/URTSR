import os


def delete(path: str) -> None:
    # Loop through all files in the directory
    for filename in os.listdir(path):
        if '.exr' in filename:
            # Check if 'velocity' is not in the filename
            if 'velocity.' not in filename:
                # Construct the full file path
                file_path = os.path.join(path, filename)
                # Remove the file
                os.remove(file_path)
                print(f"Deleted {file_path}")


def main() -> None:
    path = "LR_new/02"
    delete(path)


if __name__ == "__main__":
    main()
