import pathlib


def get_root_dir(cwd: pathlib.Path = pathlib.Path().resolve(), anchor="README.md") -> pathlib.Path:
    """
    Get the root directory of the project by searching for a specific anchor file. 
    i.e. find the root directory where anchor file README.md/.git is located.
    
    Args:
        cwd (pathlib.Path): Current working directory.
        anchor (str): The name of the anchor file to search for.

    Returns:
        pathlib.Path: The root directory of the project.

    Raises:
        FileNotFoundError: If the anchor file is not found in any parent directories.
    """
    # Check if the anchor file exists in the current working directory
    # If it does, return the current working directory
    # If it doesn't, check the parent directories until the anchor file is found
    if cwd.joinpath(anchor).exists():
        return cwd
    else:
        for parent in cwd.parents:
            if (parent / anchor).exists():
                return parent
    
    # If the anchor file is not found in any parent directories, raise an error
    raise FileNotFoundError(f"Anchor file '{anchor}' not found in any parent directories of {cwd}.")
