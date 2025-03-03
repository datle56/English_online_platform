from abc import ABC, abstractmethod
from asyncio import Event
from features.talk.logger import get_logger
from fastapi import WebSocket
import asyncio
import os
from edge_tts import Communicate, VoicesManager
from features.talk.utils import Singleton, timed

logger = get_logger(__name__)
class TextToSpeech(ABC):
    @abstractmethod
    @timed
    async def stream(
        self,
        text: str,
        websocket: WebSocket,
        tts_event: Event,
        voice_id: str,
        first_sentence: bool,
        language: str,
        *args,
        **kwargs
    ):
        pass

    async def generate_audio(self, *args, **kwargs):
        pass
    
logger = get_logger(__name__)

EDGE_TTS_DEFAULT_VOICE = "en-US-ChristopherNeural"


class EdgeTTS(Singleton, TextToSpeech):
    def __init__(self):
        super().__init__()
        logger.info("Initializing [EdgeTTS] voices...")

    @timed
    async def stream(
        self,
        text,
        websocket,
        tts_event: asyncio.Event,
        voice_id="",
        first_sentence=False,
        language="en-US",
        *args,
        **kwargs
    ) -> None:
        voices = await VoicesManager.create()
        try:
            voice = voices.find(ShortName=voice_id)[0]
        except IndexError:
            voice = voices.find(ShortName=EDGE_TTS_DEFAULT_VOICE)[0]
        communicate = Communicate(text, voice["Name"], rate="+20%")
        messages = []
        async for message in communicate.stream():
            if message["type"] == "audio":
                # Choose to accmulate the audio data because
                # the stream packets are broken when playback.
                messages.extend(message["data"])
        await websocket.send_bytes(bytes(messages))

    async def generate_audio(self, text, voice_id="", language="en-US") -> bytes:
        voices = await VoicesManager.create()
        voice = voices.find(ShortName=voice_id)[0]
        communicate = Communicate(text, voice["Name"], rate="+20%")
        messages = []
        async for message in communicate.stream():
            if message["type"] == "audio":
                messages.extend(message["data"])
        return bytes(messages)
