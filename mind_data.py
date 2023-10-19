import os
import pandas as pd
import json
from utils.download_utils import download_path, maybe_download, unzip_file



URL_MIND_LARGE_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip"
)
URL_MIND_LARGE_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip"
)
URL_MIND_SMALL_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
)
URL_MIND_SMALL_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
)
URL_MIND_DEMO_TRAIN = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip"
)
URL_MIND_DEMO_VALID = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
)
URL_MIND_DEMO_UTILS = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_utils.zip"
)

URL_MIND = {
    "large": (URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID),
    "small": (URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID),
    "demo": (URL_MIND_DEMO_TRAIN, URL_MIND_DEMO_VALID),
}

def download_mind(size="small", dest_path=None):
    """Download MIND dataset

    Args:
        size (str): Dataset size. One of ["small", "large"]
        dest_path (str): Download path. If path is None, it will download the dataset on a temporal path

    Returns:
        str, str: Path to train and validation sets.
    """
    size_options = ["small", "large", "demo"]
    if size not in size_options:
        raise ValueError(f"Wrong size option, available options are {size_options}")
    url_train, url_valid = URL_MIND[size]
    with download_path(dest_path) as path:
        train_path = maybe_download(url=url_train, work_directory=path)
        valid_path = maybe_download(url=url_valid, work_directory=path)
    return train_path, valid_path

def extract_mind(
    train_zip,
    valid_zip,
    train_folder="train",
    valid_folder="valid",
    clean_zip_file=True,
):
    """Extract MIND dataset

    Args:
        train_zip (str): Path to train zip file
        valid_zip (str): Path to valid zip file
        train_folder (str): Destination forder for train set
        valid_folder (str): Destination forder for validation set

    Returns:
        str, str: Train and validation folders
    """
    dir_folder = os.path.dirname(train_zip)
    train_path = os.path.join(dir_folder, train_folder)
    valid_path = os.path.join(dir_folder, valid_folder)
    unzip_file(train_zip, train_path, clean_zip_file=clean_zip_file)
    unzip_file(valid_zip, valid_path, clean_zip_file=clean_zip_file)
    return train_path, valid_path


def download_extract_mind(size="small", dest_path=None):
    train_zip_path, valid_zip_path = download_mind(size, dest_path)
    train_path, valid_path = extract_mind(train_zip=train_zip_path,valid_zip=valid_zip_path, clean_zip_file=False)
    # Precess data, convert .tsv to .json  format
    json_data_path = []
    for data_path in [train_path, valid_path]:
        df_path = os.path.join(data_path, "news.tsv")
        df = pd.read_table(df_path, names=['newid', 'vertical', 'subvertical', 'title','abstract', 'url', 'entities in title', 'entities in abstract'],
                            usecols = ['newid', 'title'], header=None)
        data = []
        for row in df.itertuples(index=False):
            item_data = {}
            item_data['id'] = row.newid
            item_data['title'] = row.title.split(' ')
            data.append(item_data)
        json_file_path = os.path.join(data_path, "news.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(data,json_file)
        json_data_path.append(json_file_path)

        df_path = os.path.join(data_path, "behaviors.tsv")
        df = pd.read_table(df_path, names=['Impression ID','user_id', 'time_stamp', 'clickHist', 'ImpreLog'],
                            usecols = ['user_id', 'ImpreLog'], header=None)
        data = []
        for row in df.itertuples(index=False):
            item_data = {}
            item_data['user_id'] = row.user_id
            item_data['push'] = [clicked_new[:-2] for clicked_new in row.ImpreLog.split(' ') if clicked_new[-1] == "1" ]
            data.append(item_data)
        json_file_path = os.path.join(data_path, "behaviors.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(data,json_file)
        json_data_path.append(json_file_path)
    return json_data_path

if __name__ == '__main__':
    [train_news_data, train_users_data, valid_news_data, valid_users_data] = download_extract_mind(size = "small", dest_path="./datasets")

    