# Machine Learning tests

The goal is to reliably predict the retailer name based on receipt ocr raw textual data. 
The raw retailer name comes on the `establishment` inside the `rawData` field. 
The values in `establishment` may came in different flavors (i.e. Boots vs Boots UK) and we want that the predicted value be unique (Boots).

This is a classical Text Classification (a.k.a. text categorization or text tagging) problem which is the process of assigning tags or categories to text according to its content.
In order to train our prediction model I used the `retailName` in the provided dataset as labels and the raw `establishment` field as the input features.

## Install requirements

    pip3.6 install -r requirements.txt 

## Jupyter notebook to analyse the dataset

I used the following jupyter notebook to anayse the dataset and create the
    jupyter notebook notebook/scenario_1.ipynb

## Train and save the model

    ./1-create-model.sh
or

    python3.6 src/modeltrainer/model_trainer.py -i receipt_data.csv 

## Service to predict retailer name from OCR raw data

   Run predictor service

### Run gunicorn server locally:

PYTHONPATH=src/retailerpredictor gunicorn -c gunicorn.cfg app:app 
    
### Run gunicorn server with docker:

    ./2-build-docker-image.sh
    ./3-start-dockercontainer.sh

