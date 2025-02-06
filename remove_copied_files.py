import os

def remove_copie_files(folder_path):
    try:
        files = os.listdir(folder_path)

        for file_name in files:
            if "Capture" in file_name:
                file_path = os.path.join(folder_path, file_name)

                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f'Removed: {file_name}')
        print("All 'copy' files have been removed.")
    except Exception as e:
        print(f"An error occurred: {e}")

folder_path = 'C:\\Users\\DELL\\Pictures'
remove_copie_files(folder_path)
