import os
import boto3 
import sys
import progressbar

def upload(data_dir, key):
    s3 = boto3.resource('s3')

    files = os.listdir(data_dir)
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i, f in enumerate(files):
            data = open(data_dir + f, 'rb')
            s3_object_key = key + "/" + f
            s3.Bucket('fix-it-in-post').put_object(Key=s3_object_key, Body=data)
            bar.update(i)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    key = data_dir.split("/")
    key = key[len(key) - 2]
    upload(data_dir, key)