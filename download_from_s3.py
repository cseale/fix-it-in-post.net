import boto3

s3 = boto3.resource('s3')
my_bucket = s3.Bucket('fix-it-in-post')

for object in my_bucket.objects.all():
    print(object)