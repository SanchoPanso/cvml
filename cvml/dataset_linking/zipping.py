import os
import zipfile


def unzip_to(src_path: str, dst_path: str):
    with zipfile.ZipFile(src_path, 'r') as zip_ref:
        zip_ref.extractall(dst_path)


def zip_to(src_path: str, dst_path: str):
    pass
