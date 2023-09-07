#!/bin/bash

cp ./data/preprocessed/* ./data/upload/
cp -r data/model/* ./data/upload/
cp -r ./src/ ./data/upload/

kaggle datasets version -r "zip" -p ./data/upload/ -m "Update" 
