from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from memories.dl_logic.model import initialize_model, predict
from memories.dl_logic.transformers_preprocessing import clean_df, tokenize_data, padding
from pydantic import BaseModel

app = FastAPI()

# Allow all requests (optional, good for development purposes)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
 )
app.state.model = initialize_model()

class TextInput(BaseModel):
    text: str

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_memory')
async def receive_history(input_data: TextInput):

    # Preprocess the input text
    clean_dataset = clean_df()  # Your data cleaning logic
    '''
    Not sure about this
    '''
    tokenized = tokenize_data(input_data)  # Tokenize the text
    padded_data = padding(tokenized)  # Apply padding

    # Use the model for prediction
    pred = predict(app.state.model, padded_data.text)

    return pred
