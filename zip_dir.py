import os
import zipfile

def zip_selected_files(source_dir, zip_filename):
    # File extensions to include
    include_exts = ('.py', '.txt', '.json', '.jsonl')

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Skip the 'artifacts' directory entirely
            if 'artifacts' in dirs:
                dirs.remove('artifacts')
            if '.venv' in dirs:
                dirs.remove('.venv')   
            if 'indexes' in dirs:
                dirs.remove('indexes')         
            # if 'evidence_cache' in dirs:
            #     dirs.remove('evidence_cache')                                             

            for file in files:
                if file.endswith(include_exts):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, start=source_dir)
                    zipf.write(full_path, rel_path)
                    print(f"Added: {rel_path}")

    print(f"\nCreated ZIP: {zip_filename}")

# Example usage
zip_selected_files("fakecheck", "fakecheck.zip")

