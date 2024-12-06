from fastapi import APIRouter, Depends
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
# import lambdaTTS
from . import lambdaSpeechToScore
# import lambdaGetSample
import eng_to_ipa as ipa
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from session.dependencies import get_db
from sqlalchemy.orm import Session
from session.auth import oauth2_scheme, Token, TokenData, create_access_token, get_current_user
from session.database import User, History
from datetime import datetime


class TextRequest(BaseModel):
    text: str

router = APIRouter(prefix="", tags=["Pronouce"])

@router.post("/GetIPA")
async def get_accuracy_from_recorded_audio(request: TextRequest):
    text = request.text
    ipa_text = ipa.convert(text)
    ipa_text = ipa_text.replace("ˈ", "")
    return JSONResponse(content={"text": text, "ipa": ipa_text})

@router.post("/GetAccuracyFromRecordedAudio")
async def get_accuracy_from_recorded_audio(request: Request,current_user: User = Depends(get_current_user),db: Session = Depends(get_db)
):
    event = {'body': json.dumps(await request.json())}
    # print(event)
    lambda_correct_output = lambdaSpeechToScore.lambda_handler(event, [])
    # print(lambda_correct_output)
    real_text = lambda_correct_output.get("real_text", "")
    matching_result = lambda_correct_output.get("matching_result", "")
    username = current_user.username

    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"error": "User not found"}
    
    # Insert a new record into the History table
    new_history = History(
        user_id=user.id,
        feature="Pronunciation",
        input_data=real_text,  # Input data is the real_text provided by the user
        output_data=matching_result,
        created_at = datetime.utcnow(),
    )

    db.add(new_history)
    db.commit()
    db.refresh(new_history)


    return lambda_correct_output

# @router.post("/GetIPA")
# async def get_accuracy_from_recorded_audio(request: TextRequest):
#     text = request.text
#     ipa_text = ipa.convert(text)
#     ipa_text = ipa_text.replace("ˈ", "")
#     return JSONResponse(content={"text": text, "ipa": ipa_text})