import os
import boto3 
from process import process_audio, get_directory_name

DIR = "./data/edinburgh-noisy-speech-db"

def upload_files_to_s3(data_dir, key):
    s3 = boto3.resource('s3')

    for f in os.listdir(DIR + data_dir):
        data = open(DIR + data_dir + f[0], 'rb')
        s3.Bucket('fix-it-in-post').put_object(Key=data_dir + f[0], Body=data)
