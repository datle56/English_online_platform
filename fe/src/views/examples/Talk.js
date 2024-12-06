import React, { useState, useEffect, useRef } from 'react';
import './chat.css';

const ChatComponent = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const audioQueue = useRef([]); // Queue for audio data
  const textQueue = useRef([]); // Queue for text data
  const messagesDiv = useRef(null);
  const isPlayingAudio = useRef(false); // Flag to check if audio is playing
  const isTypingText = useRef(false); // Flag to check if typing effect is active
  const ws = useRef(null);
  const pingInterval = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const messageBuffer = useRef(''); // Buffer for accumulating incoming text

  useEffect(() => {
    const token = localStorage.getItem('token'); // Hoặc lấy token từ Redux store hoặc Context
  
    if (!token) {
      console.error("No token found");
      return;
    }
  
    // Thêm token vào URL của WebSocket
    const url = `ws://localhost:8000/chat/ws/22?token=${token}`;
    ws.current = new WebSocket(url);
    ws.current.binaryType = 'arraybuffer';
  
    ws.current.onopen = () => {
      console.log('[WebSocket] Connected to server');
      pingInterval.current = setInterval(() => {
        ws.current.send('ping');
      }, 30000);
    };
  
    ws.current.onmessage = (event) => {
      console.log('[WebSocket] Received message:', event.data);
  
      if (typeof event.data === 'string') {
        handleTextMessage(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        enqueueAudio(event.data);
      } else {
        console.warn('[WebSocket] Unknown message type:', event.data);
      }
    };
  
    ws.current.onclose = () => {
      console.log('[WebSocket] Connection closed');
      clearInterval(pingInterval.current);
    };
  
    ws.current.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };
  
    return () => {
      clearInterval(pingInterval.current);
      ws.current.close();
    };
  }, []);
  

  const handleTextMessage = (message) => {
    if (message.startsWith('[start]')) {
      // Reset the message buffer
      messageBuffer.current = '';
      // Display the starting message immediately without queuing it
      const initialMessage = message.substring(7).trim();
      addMessage(initialMessage, 'agent');
    } else if (message === '[end start]') {
      // Reset the message buffer when receiving '[end start]'
      messageBuffer.current = '';
    } else if (message.startsWith('[end=')) {
      // Use only the content inside '[end=...]' as the complete message
      const completeMessage = message.substring(5, message.length - 1);
      messageBuffer.current = ''; // Clear the buffer after processing
      enqueueText(completeMessage);
    } else if (message.startsWith('[+]')) {
      // This is the transcribed user message
      const userMessage = message.substring(3).trim();
      // Optionally, remove 'You said:' prefix if present
      if (userMessage.startsWith('You said:')) {
        const transcribedText = userMessage.substring(9).trim();
        addMessage(transcribedText, 'you');
      } else {
        addMessage(userMessage, 'you');
      }
    } else {
      // Append any other parts to the buffer
      messageBuffer.current += message;
    }
  };

  const enqueueText = (text) => {
    textQueue.current.push(text);
    if (!isTypingText.current) processTextQueue();
  };

  const processTextQueue = () => {
    if (textQueue.current.length === 0) {
      isTypingText.current = false;
      return;
    }

    isTypingText.current = true;
    const text = textQueue.current.shift();

    // Add a new message with empty text and typing: true
    const newMessage = {
      id: Date.now(),
      text: '',
      sender: 'agent',
      typing: true,
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);

    // Start typing effect with the message id
    startTypingEffect(text, newMessage.id, 20, () => {
      isTypingText.current = false;
      processTextQueue();
    });
  };

  const startTypingEffect = (text, messageId, delay = 1, callback) => {
    let index = 0;

    const typeCharacter = () => {
      if (index < text.length) {
        const currentText = text.substring(0, index );
        // Update the message in the state
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === messageId ? { ...msg, text: currentText } : msg
          )
        );
        index++;
        if (messagesDiv.current) {
          messagesDiv.current.scrollTop = messagesDiv.current.scrollHeight;
        }
        setTimeout(typeCharacter, delay);
      } else {
        // Once typing is complete, update the message
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === messageId ? { ...msg, typing: false } : msg
          )
        );
        if (callback) callback();
      }
    };

    typeCharacter();
  };

  const enqueueAudio = (audioData) => {
    audioQueue.current.push(audioData);
    if (!isPlayingAudio.current) playNextAudio();
  };

  const playNextAudio = () => {
    if (audioQueue.current.length === 0) {
      isPlayingAudio.current = false;
      return;
    }

    isPlayingAudio.current = true;
    const audioData = audioQueue.current.shift();
    playAudio(audioData);
  };

  const playAudio = (audioData) => {
    const blob = new Blob([audioData], { type: 'audio/webm; codecs=opus' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);

    audio.onended = () => {
      URL.revokeObjectURL(url);
      isPlayingAudio.current = false;
      playNextAudio();
    };

    audio.play().catch((err) => {
      console.error('[Audio] Playback error:', err);
      isPlayingAudio.current = false;
      playNextAudio();
    });
  };

  const addMessage = (message, sender) => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Date.now(), text: message, sender, typing: false },
    ]);
    setTimeout(() => {
      if (messagesDiv.current) {
        messagesDiv.current.scrollTop = messagesDiv.current.scrollHeight;
      }
    }, 0);
  };

  const sendMessage = () => {
    if (inputValue.trim()) {
      addMessage(inputValue, 'you');
      ws.current.send(inputValue);
      setInputValue('');
    }
  };

  const handleAudioRecording = async () => {
    if (isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        mediaRecorder.current = new MediaRecorder(stream);
        mediaRecorder.current.ondataavailable = (event) =>
          audioChunks.current.push(event.data);
        mediaRecorder.current.onstop = () => {
          const audioBlob = new Blob(audioChunks.current, {
            type: 'audio/webm; codecs=opus',
          });
          ws.current.send(audioBlob);
          audioChunks.current = [];
        };
        mediaRecorder.current.start();
        setIsRecording(true);
      } catch (error) {
        console.error('[Audio] Error accessing microphone:', error);
      }
    }
  };

  return (
    <div className="chat-root">
      <div className="chat-container shadow">
        {/* <div className="header">
          <h1 className="header-title">Talk with AI</h1>
        </div> */}
        <div ref={messagesDiv} className="messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message..."
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button onClick={sendMessage}>➤</button>
          <button className="voice-button" onClick={handleAudioRecording}>
            {isRecording ? '⏹️' : '🎤'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatComponent;
