# fix-it-in-post.net

Project based of this template: https://github.com/victoresque/pytorch-template#pytorch-template-project

### Creating Sagemaker Instances
- Connect to fix-it-in-post.net repo
- Remember to use git credentials and access keys
- Create configuration to run each time notebook is start to install tensorboard on startup, using `./sagemaker_scripts/setup.sh`

### Processing
Processes the files. Passing `all` will process all the data, by default it will process 512 samples
```
python process.py [all]
```

### Training
The following command will run the training script. If the data exists locally or on S3, the data will be pulled from those locations. If not, the data will be processed locally instead.
```
python train.py --config config/baseline_fc.json
```

### Upload to S3
If you have data processed locally, it would be good to store on S3 for others to use without having to reprocess the data. You can store the data on S3 as such:
```
python upload.py ./data/processed/edinburgh-noisy-speech-db/w256o75sr8000n8/
```
This will store the data under the key `w256o75sr8000n8/sample.0.pkl` in the bucket `fix-it-in-post`