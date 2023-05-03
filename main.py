import os
import openai
from dotenv import load_dotenv
from llama_index import download_loader
from gpt_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def getDocsFromGoogleDrive():
    GoogleDriveReader = download_loader('GoogleDriveReader')
    loader = GoogleDriveReader()
    documents = loader.load_data(folder_id='1iHsPxQVVMoAlR2i0myCBbwE9UzKcfuB1')

    return documents


def createDatasetIndex():
    documents = getDocsFromGoogleDrive()

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Construct a simple vector index
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    # Save your index to a index.json file
    index.save_to_disk('index.json')

    # Load the index from your saved index.json file
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    return index


def getDatasetIndex():
    if (os.path.exists('index.json')):
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        return index
    
    index = createDatasetIndex()
    return index
