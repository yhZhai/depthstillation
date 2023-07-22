import argparse
from pathlib import Path
from tqdm import tqdm

def rename_files_in_folder(folder_path: str):
    # Convert string path to a Path object
    folder_path = Path(folder_path)
    files_to_rename = [f for f in folder_path.iterdir() if f.is_file() and "_00." in f.name]

    # Loop through all files in the specified folder with a progress bar
    for file_path in tqdm(files_to_rename, desc="Renaming", unit="file"):
        # Replace "_00" with an empty string
        new_name = file_path.name.replace("_00.", ".")
        new_path = file_path.parent / new_name
        
        # Rename the file
        file_path.rename(new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files in a directory by removing '_00' before the file extension.")
    parser.add_argument("directory", type=str, help="Path to the directory containing files to rename.")
    args = parser.parse_args()

    rename_files_in_folder(args.directory)
