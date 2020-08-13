import os
import sys
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

thisdir = os.path.dirname(os.path.realpath(__file__))

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"file exists in {output_path}. specify overwrite=True if intended")

def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path) as zip:
        zip.extractall(target_dir)

url = "https://syncandshare.lrz.de/dl/fi9CbfDXCJHarTS1K7znjtrp/sen12ms-rgb-4ways-2shot.zip"
if not os.path.exists(thisdir + "/static/sen12ms-rgb-4ways-2shot.zip"):
    download_file(url, thisdir + "/static/sen12ms-rgb-4ways-2shot.zip", overwrite=False)
    unzip(thisdir + "/static/sen12ms-rgb-4ways-2shot.zip", thisdir + "/static/")
