"""
Filename: remove_unused_libs_in_requirements_txt.py

Module to identify and remove unused packages from a project’s virtual environment.

Steps to find and clean up unused dependencies:

1. **Activate Virtual Environment** (if not already active):
    - **PowerShell**:
      ```powershell
      .\path\to\env\Scripts\Activate.ps1
      ```
    - **Bash** (Linux/macOS):
      ```bash
      source path/to/env/bin/activate
      ```

2. **Generate List of All Installed Packages**:
    - Save a list of all packages currently installed in the environment to 
    a file named `all_packages.txt`.
    - **PowerShell** or **Bash**:
      ```powershell
      pip freeze > all_packages.txt
      ```

3. **Generate a Fresh `requirements.txt` File with Only Used Packages**:
    - Use `pipreqs` to scan the project’s directory and generate a `requirements.txt` file that 
    includes only packages directly imported in your code.
    - **Install pipreqs (if not already installed)**:
      ```powershell
      pip install pipreqs
      ```
    - **Generate requirements.txt (replace `/path/to/your/project` with your project directory)**:
      ```powershell
      pipreqs /path/to/your/project --force
      ```

4. **Run Python Script to Remove Unused Packages**:
    - The Python script will read `all_packages.txt` and `requirements.txt`, 
    identify any packages listed in `all_packages.txt` but not in `requirements.txt`, 
    and uninstall those unused packages.
    - Ensure this script is in the project’s root directory where both `all_packages.txt` 
    and `requirements.txt` are located.
    - **Run the Script**:
      ```powershell
      python remove_unwanted_packages.py
      ```

This module helps automate the process of identifying and removing unused packages 
to keep the virtual environment clean and reduce dependency bloat.

*Only run this after activated your virtual environment.
"""

import os
import chardet


# Set the root directory to the directory where this script is located
root_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_dir)


# Detect encoding
with open("all_packages.txt", "rb") as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)["encoding"]

# Read all installed packages
with open("all_packages.txt", "r", encoding=encoding) as all_file:
    all_packages = set(line.strip().split("==")[0] for line in all_file if "==" in line)

# Read required packages
with open("requirements.txt", "r") as req_file:
    required_packages = set(
        line.strip().split("==")[0] for line in req_file if "==" in line
    )

# Find unused packages
unused_packages = all_packages - required_packages
print("Unused packages:", unused_packages)

# Confirm before uninstalling
if unused_packages:
    confirm = input("Do you want to uninstall these unused packages? (y/n): ")
    if confirm.lower() == "y":
        for package in unused_packages:
            os.system(f"pip uninstall -y {package}")
        print("Unused packages uninstalled.")
    else:
        print("Uninstallation canceled.")
else:
    print("No unused packages found.")
