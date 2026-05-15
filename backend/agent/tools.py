# from typing import Tuple
# import subprocess

# tools.py
from langchain_core.tools import tool

_VFS: dict = {}

def init_vfs():
    global _VFS
    _VFS.clear()  # Clear IN PLACE — don't reassign

def get_vfs() -> dict:
    return _VFS

@tool
def write_file(path: str, content: str) -> str:
    """Writes code content to a file at the specified path within the virtual file system."""
    _VFS[path] = content
    return f"WROTE to memory: {path}"

@tool
def read_file(path: str) -> str:
    """Reads code content from a file at the specified path from the virtual file system."""
    return _VFS.get(path, "File does not exist yet.")

@tool
def list_files(directory: str = ".") -> str:
    """Lists all files currently in the virtual file system."""
    return "\n".join(_VFS.keys()) if _VFS else "No files found."