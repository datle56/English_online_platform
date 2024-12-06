from fastapi import FastAPI
from pydantic import BaseModel
from .gector import predict, load_verb_dict
from transformers import AutoTokenizer, AutoModel
import torch
from .grammar_check import llm_feedback
import os
# Load resources and model at startup
encode, decode = load_verb_dict(os.path.join(os.path.dirname(__file__), 'data', 'verb-form-vocab.txt'))
model = torch.load(os.path.join(os.path.dirname(__file__), 'gector-deberta-v3.pth'))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), 'token'))
from fastapi import APIRouter,Depends
from session.auth import oauth2_scheme, Token, TokenData, create_access_token, get_current_user
from session.database import User, History
from datetime import datetime
from sqlalchemy.orm import Session
from session.dependencies import get_db
from difflib import SequenceMatcher


if torch.cuda.is_available():
    model.cuda()

router = APIRouter(prefix="", tags=["Grammar"])

# Define input and output models
class CorrectionRequest(BaseModel):
    text: str

class CorrectionResponse(BaseModel):
    explanation : str
    corrected_sentence: str
    mix_sentence : str
    colour : str

@router.post("/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest, current_user: User = Depends(get_current_user),db: Session = Depends(get_db)):
    # Convert the input text (str) into a list with one sentence
    srcs = [request.text]

    predict_args = {
        'model': model,
        'tokenizer': tokenizer,
        'srcs': srcs,
        'encode': encode,
        'decode': decode,
        'keep_confidence': 0,
        'min_error_prob': 0,
        'batch_size': 1,
        'n_iteration': 5
    }

    # Generate corrected sentences
    final_corrected_sents = predict(**predict_args)
    
    # Join the result back into a single string
    corrected_text = final_corrected_sents[0] if final_corrected_sents else ""
    print(corrected_text)
    explanation, corrected_sentence = llm_feedback(input_sentence= request.text, output_sentence= corrected_text)


    #add into database 
    username = current_user.username
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"error": "User not found"}

    input_text = srcs[0]
    print(input_text)
    new_history = History(
        user_id=user.id,
        feature="Grammar",
        input_data=input_text,  # Input data is the real_text provided by the user
        output_data=corrected_sentence,
        created_at = datetime.utcnow(),
    )

    db.add(new_history)
    db.commit()
    db.refresh(new_history)

    source_sent = input_text
    target_sent = corrected_sentence


    source_tokens = source_sent.split()
    target_tokens = target_sent.split()
    matcher = SequenceMatcher(None, source_tokens, target_tokens)
    diffs = list(matcher.get_opcodes())
    text=''
    colour=''
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if tag == 'equal':
            #Keep
            for i in source_tokens[i1:i2]:
                text=text + i + ' '
                for ii in i:
                    colour+='1'
                colour+=' '
        else: 
            #Delete
            for i in source_tokens[i1:i2]:
                text=text + i + ' '
                for ii in i:
                    colour+='2'
                colour+=' '
            #Add
            for j in target_tokens[j1:j2]:
                text=text + j + ' '
                for jj in j:
                    colour+='3'
                colour+=' '
    print(text)
    print(colour)



    return CorrectionResponse(explanation=explanation, corrected_sentence=corrected_sentence, mix_sentence =text, colour = colour )

# Run with: uvicorn filename:app --reload
