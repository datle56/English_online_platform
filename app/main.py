from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from features.user.router import router as user_router
from features.talk.routers.chat import router as chat_router
from features.grammar.router import router as grammer_router
from features.pronounce.router import router as pronouce_router

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('features\\grammar\\gector')))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(user_router, prefix="", tags=["User"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(grammer_router, prefix="/grammar")
app.include_router(pronouce_router, prefix="")


