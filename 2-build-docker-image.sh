#!/bin/bash

docker rm -f wevat-predictor
docker rmi -f wevat-predictor
docker image build --tag wevat-predictor .
