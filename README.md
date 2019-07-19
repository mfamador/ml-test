# Machine Learning tests

The goal is to reliably predict the retailer name based on receipt ocr raw textual data. 
The raw retailer name comes on the `establishment` inside the `rawData` field. 
The values in `establishment` may came in different flavors (i.e. Boots vs Boots UK) and we want that the predicted value be unique (Boots).

## Install requirements

    pip3.6 install -r requirements.txt 

## Jupyter notebooks to analyse the dataset and create the model for the different scenarios:
    
[notebook/scenario_1.ipynb](scenario 1)
    
[notebook/scenario_3.ipynb](scenario 3)

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

