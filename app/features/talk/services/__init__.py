import os
from typing import Optional
from features.talk.services.base import SpeechToText, TextToSpeech
from features.talk.services.stt import Whisper
from features.talk.services.tts import EdgeTTS

def get_speech_to_text() -> SpeechToText:
    Whisper.initialize(use="api")
    return Whisper.get_instance()

def get_text_to_speech() -> TextToSpeech:


    EdgeTTS.initialize()
    return EdgeTTS.get_instance()
