#!/bin/bash

MESSAGE=$1

cp ./data/preprocessing/* ./data/upload/
cp ./data/models/* ./data/upload/
cp -r ./src/ ./data/upload/

kaggle datasets version -r "zip" -p ./data/upload/ -m $MESSAGE 
