import shutil
import io
import subprocess

if __name__ == '__main__':
    current_version = "0.0.1"
    new_version = "0.0.1"
    with io.open(f"src/llm_recovery/__version__.py", 'r', encoding='utf-8') as f:
        version_file_contents = f.read()
    with io.open(f"src/llm_recovery/__version__.py", 'w', encoding='utf-8') as f:
        version_file_contents = version_file_contents.replace(current_version, new_version)
        f.write(version_file_contents + "\n")

    # Delete old build directories
    print("Delete old build directories")
    shutil.rmtree(f"dist", ignore_errors=True)

    # Build
    p = subprocess.Popen(f"python -m build", stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    exit_code = p.wait()
    output = str(output)
    err = str(err)
    if exit_code == 0:
        print(f"built successfully")
    else:
        print(f"There was an error building; exit code: {exit_code}")
        print(output)
        print(err)

    # Push
    print("Push to PyPi")
    p = subprocess.Popen(f"python -m twine upload --config-file ~/.pypirc dist/*",
                         stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    exit_code = p.wait()
    output = str(output)
    err = str(err)
    if exit_code == 0:
        print(f"Successfully uploaded to PyPi")
    else:
        print(f"There was an error uploading to PyPi; exit code: {exit_code}")
        print(output)
        print(err)
