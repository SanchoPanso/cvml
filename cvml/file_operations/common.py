import os
import shutil


def mv_dir(src: str, dst: str):
    # if a directory of the same name does not exist in the destination, we can simply rename the directory
    # to a different path, and it will be moved -- it will disappear from the source path and appear in the destination
    # path instantaneously, without any files being copied.
    try:
        os.rename(src, dst)
        return
    except FileExistsError:
        # if a directory of the same name already exists, we must merge them.  This is what the algorithm below does.
        pass
    for root, dirs, files in os.walk(src):
        dest_root = os.path.join(dst, os.path.relpath(root, src))
        done = []
        for dir_ in dirs:
            try:
                os.rename(os.path.join(root, dir_), os.path.join(dest_root, dir_))
                done.append(dir_)
            except FileExistsError:
                pass
        # tell os.walk() not to recurse into subdirectories we've already moved.  see the documentation on os.walk()
        # for why this works: https://docs.python.org/3/library/os.html#os.walk
        # lists can't be modified during iteration, so we have to put all the items we want to remove from the list
        # into a second list, and then remove them after the loop.
        for dir_ in done:
            dirs.remove(dir_)
        # move files.  os.replace() is a bit like os.rename() but if there's an existing file in the destination with
        # the same name, it will be deleted and replaced with the source file without prompting the user.  It doesn't
        # work on directories, so we only use it for files.
        # You may want to change this to os.rename() and surround it with a try/except FileExistsError if you
        # want to prompt the user to overwrite files.
        for file in files:
            os.replace(os.path.join(root, file), os.path.join(dest_root, file))
    # clean up after ourselves.
    # Directories we were able to successfully move just by renaming them (directories that didn't exist in the
    # destination already) have already disappeared from the source.  Directories we had to merge are still there in
    # the source, but their contents were moved.  os.rmdir() will fail unless the directory is already empty.
    for root, dirs, files in os.walk(src, topdown=False):
        os.rmdir(root)


def rm_dir(path):
    shutil.rmtree(path)


def mv_files_from_dir(src_path: str, dst_path: str, delete_src_dir: bool = True):
    pass



