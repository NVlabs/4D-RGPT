from pathlib import Path

def is_subdirectory(sub_dir: str, parent_dir: str) -> bool:
    sub_path = Path(sub_dir).resolve()
    parent_path = Path(parent_dir).resolve()
    return parent_path in sub_path.parents

def get_relative_path(sub_dir: str, parent_dir: str) -> str | None:
    parent_path = Path(parent_dir).resolve()
    sub_path = Path(sub_dir).resolve()

    if parent_path in sub_path.parents:
        rel_path = sub_path.relative_to(parent_path)
        return str(rel_path)  # Returns path difference if sub_dir is inside parent_dir
    else:
        return None  # Not a subdirectory