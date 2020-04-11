#!/bin/bash
app="russian-bert-absa-api"
docker run --name russian-bert-absa \
      --mount "type=bind,source=$(pwd)/api,target=/app/api" \
      --mount "type=bind,source=$(pwd)/notebooks,target=/app/notebooks" \
      -d -p 8080:5000 ${app}