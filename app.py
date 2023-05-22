import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import createDatasetIndex, createDatasetIndexFromGoogle, getDatasetIndex, getDatasetIndexFromGoogle, getPrompt
from typing import Union

logger = logging.getLogger('app')

class QuestionData(BaseModel):
    prompt: str

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info('App has been started')

@app.post("/")
async def create_prompt(data: QuestionData, source: Union[str, None] = None):
    prompt = getPrompt(data.prompt)
    logger.info(prompt)

    if source == 'google':
        index = getDatasetIndexFromGoogle()
        return index.query(prompt)
    elif source == 'pdf':
        index = getDatasetIndex()
        return index.query(prompt)

@app.post("/sync-dataset")
async def sync_dataset():
    createDatasetIndex()
    logger.info('Sync dataset')
    return

@app.post("/sync-dataset-google")
async def sync_dataset():
    createDatasetIndexFromGoogle()
    logger.info('Sync dataset from Google')
    return
