import glob
import os
import sys

# The conflicting file name
filename = "field_behavior_pb2.py"
# The directory inside the packages
path_fragment = os.path.join("google", "api")

found_files = []

print(f"Searching for {filename} in your Python path...\n")

# sys.path contains all the locations where python looks for modules
for path in sys.path:
    # We only care about site-packages where installed libraries live
    if "site-packages" in path:
        search_pattern = os.path.join(path, path_fragment, filename)
        # Using glob to find the file
        results = glob.glob(search_pattern, recursive=True)
        if results:
            found_files.extend(results)

if len(found_files) > 1:
    print("CONFLICT CONFIRMED. Found multiple copies of the file:\n")
    for f in found_files:
        print(f"- {f}")
    print("\nThis is the cause of the TypeError.")
else:
    print("No conflict found. The issue may be different.")
