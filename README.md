# mobile-price-prediction

## Software and account Required

1.[Github Account](https://github.com)
2.[AWS Account]()
3.[VS Code IDE](https://code.visualstudio.com/download)
4.[GIT cli](https://git-scm.com/downloads)42A5-E86E

Creating conda environment
```
conda create -p venv python==3.8 -y
```

```
conda activate venv/
```
OR
```
conda activate venv
```

Making important installations
```
pip install -r requirements.txt
```

Note: Before making commit to github mention venv/ (virtual environment/) in .gitignore to avoid error due to large file size.

To Add files to git
```
git add .
```

OR
```
git add filename or foldername
```

>Note:To ignore file or folder from git we can write name of file/folder in .gitignore.file

To check the git status
```
git log
```

To create version/commit all changes by git
```
git commit -m "message" 
```

To send version/changes to github
```
git push origin main
```

To check remote url
```
git remote -v
```

Refer AWS sagemaker SDK model training documentation:https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb
