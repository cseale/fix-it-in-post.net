import boto3 

s3 = boto3.resource('s3')

data = open('./data/processed/edinburgh-noisy-speech-db/train.128.pkl', 'rb')
s3.Bucket('fix-it-in-post').put_object(Key='train.128', Body=data)