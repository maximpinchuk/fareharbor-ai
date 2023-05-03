import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import createDatasetIndex, getDatasetIndex

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
async def create_prompt(data: QuestionData):
    logger.info(data.prompt)
    index = getDatasetIndex()
    return index.query(data.prompt)

@app.post("/sync-dataset")
async def sync_dataset():
    createDatasetIndex()
    logger.info('Sync dataset')
    return
