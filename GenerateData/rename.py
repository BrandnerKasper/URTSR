import os


def rename_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in files:
        # Check if the file name contains "FinalImage."
        if "FinalImage." in file_name:
            # Construct the full file path
            old_file_path = os.path.join(folder_path, file_name)

            # New file name with "FinalImage" removed
            new_file_name = file_name.replace("FinalImage.", "")

            try:
                # Construct the full new file path
                new_file_path = os.path.join(folder_path, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{file_name}' to '{new_file_name}' in {folder_path}")
            except Exception as e:
                print(f"Error occurred while renaming '{file_name}': {e}")

        # Check if the file name contains "FinalImage"
        if "FinalImage" in file_name:
            # Construct the full file path
            old_file_path = os.path.join(folder_path, file_name)

            # New file name with "FinalImage" removed
            new_file_name = file_name.replace("FinalImage", "")

            try:
                # Construct the full new file path
                new_file_path = os.path.join(folder_path, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{file_name}' to '{new_file_name}' in {folder_path}")
            except Exception as e:
                print(f"Error occurred while renaming '{file_name}': {e}")


def main() -> None:
    # Specify the folder path
    folder_path = "LR_new/04"

    # Call the function to remove "FinalImage" from file names
    rename_files(folder_path)


if __name__ == "__main__":
    main()
