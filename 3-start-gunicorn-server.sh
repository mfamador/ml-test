#!/bin/bash

RESOURCE_PATH=resources PYTHONPATH=src/retailerpredictor gunicorn -c gunicorn.cfg app:app
