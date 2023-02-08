import os
import py7zr
import zipfile


def unzip_to(src_path: str, dst_path: str):
    with zipfile.ZipFile(src_path, 'r') as zip_ref:
        zip_ref.extractall(dst_path)


def zip_to(src_path: str, dst_path: str):
    pass


def unpack_7z(src_path: str, dst_path: str):
    with py7zr.SevenZipFile(src_path, mode='r') as z:
        z.extractall(dst_path)

def pack_7z(src_path: str, dst_path: str):
    with py7zr.SevenZipFile(src_path, 'w') as z:
        z.writeall(dst_path)

