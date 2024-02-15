#!/bin/bash

aws sso login --profile lia-001

eval "$(aws configure export-credentials --profile lia-001 --format env)"

docker-compose up
