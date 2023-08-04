# from fastapi import FastAPI
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import JSONResponse
import json
# import uvicorn
from main import simpleFilterResponse
# import sys, os

# app = FastAPI()

# @app.get("/filter")
# def filter_queryt(q: str):
# 	outVal = simpleFilterResponse(q)
# 	jsonVal = jsonable_encoder(outVal)
# 	return JSONResponse(content=jsonVal)

# @app.get("/ping")
# def ping():
# 	# Test connection without calling the model.
# 	return JSONResponse(content=jsonable_encoder({'pong': True}))

def lambda_handler(event, context):
	query = event['text']
	response = simpleFilterResponse(query)
	response = json.dumps(response)
	return {'statusCode': 200, 'body': response, 'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}}