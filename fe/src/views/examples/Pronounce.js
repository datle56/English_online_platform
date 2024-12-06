import React, { useState, useRef } from 'react';
import './chat.css';

function PronunciationApp() {
  const [inputText, setInputText] = useState('');
  const [displayText, setDisplayText] = useState('');
  const [ipaText, setIpaText] = useState('');
  const [predictedText, setPredictedText] = useState('');
  const [matchingResult, setMatchingResult] = useState('');
  const [score, setScore] = useState(0);
  const [isMicActive, setIsMicActive] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [generatedAudio, setGeneratedAudio] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioPlayerRef = useRef(null);

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleConfirm = async () => {
    // setInputText('');
    setDisplayText(inputText);

    setMatchingResult('');
    setPredictedText('');

    try {
      const response = await fetch(`http://localhost:8000/GetIPA`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      const data = await response.json();

      setIpaText(data.ipa);
    } catch (error) {
      console.error('Error fetching IPA:', error);
    }
  };

  const handleMic = async () => {
    if (!isMicActive) {
      setIsMicActive(true);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];

        mediaRecorderRef.current.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorderRef.current.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, {
            type: 'audio/ogg',
          });
          const base64Audio = await blobToBase64(audioBlob);
          await sendAudioToBackend(base64Audio);
        };

        mediaRecorderRef.current.start();
      } catch (error) {
        console.error('Error accessing microphone:', error);
      }
    } else {
      setIsMicActive(false);
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    }
  };

  const blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const sendAudioToBackend = async (base64Audio) => {
    try {
      // Retrieve the token from browser storage
      const token = localStorage.getItem('token'); // Or sessionStorage depending on your authentication method

      if (!token) {
        console.error('No authentication token found');
        return;
      }

      const response = await fetch(
        'http://localhost:8000/GetAccuracyFromRecordedAudio',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
            Authorization: `Bearer ${token}`, // Add the token to the Authorization header
          },
          body: JSON.stringify({
            title: displayText,
            base64Audio: base64Audio,
            language: 'en',
            comparison_mode: 'text',
          }),
        }
      );

      // Check if the response is ok (status in the range 200-299)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      setPredictedText(data.predicted_text);
      setMatchingResult(data.matching_result);
      setRecordedAudio(data.base64_audio);
      setScore(data.score);
    } catch (error) {
      console.error('Error sending audio to backend:', error);
      // Optional: Add user-friendly error handling
      // For example, you might want to set an error state or show a notification
    }
  };

  const handlePlayRecording = () => {
    if (recordedAudio && audioPlayerRef.current) {
      try {
        const audioSource = recordedAudio.startsWith('data:audio')
          ? recordedAudio
          : null;

        if (!audioSource) {
          console.warn('No valid audio source available.');
          return;
        }

        if (!isPlaying) {
          audioPlayerRef.current.src = audioSource;
          audioPlayerRef.current.play();
          setIsPlaying(true);

          audioPlayerRef.current.onended = () => {
            setIsPlaying(false);
          };
        } else {
          audioPlayerRef.current.pause();
          audioPlayerRef.current.currentTime = 0;
          setIsPlaying(false);
        }
      } catch (error) {
        console.error('Error playing recording:', error);
      }
    } else {
      console.warn('Audio player or recorded audio is not available.');
    }
  };

  const handlePlayGeneratedAudio = () => {
    if (generatedAudio && audioPlayerRef.current) {
      try {
        const audioSource = generatedAudio.startsWith('data:audio')
          ? generatedAudio
          : null;

        if (!audioSource) {
          console.warn('No valid audio source available.');
          return;
        }

        audioPlayerRef.current.src = audioSource;
        audioPlayerRef.current.play();
      } catch (error) {
        console.error('Error playing generated audio:', error);
      }
    } else {
      console.warn('Audio player or generated audio is not available.');
    }
  };

  const renderText = () => {
    if (!displayText) return null;

    const textWords = displayText.split(' ');
    const ipaWords = ipaText.split(' ');
    const matchingValues = matchingResult
      ? matchingResult.trim().split(' ')
      : [];

    return textWords.map((word, index) => {
      const matchingWord = matchingValues[index] || '';
      const ipaWord = ipaWords[index] || '';

      return (
        <div
          key={index}
          style={{
            display: 'inline-block',
            textAlign: 'center',
            marginRight: '15px',
          }}
        >
          <div
            style={{
              fontSize: '24px',
              fontWeight: '600',
              display: 'flex',
              justifyContent: 'center',
            }}
          >
            {word.split('').map((char, charIndex) => {
              const matchValue = matchingWord[charIndex];
              const charColor =
                matchValue === '1'
                  ? 'green'
                  : matchValue === '0'
                  ? 'red'
                  : 'black';

              return (
                <span key={charIndex} style={{ color: charColor }}>
                  {char}
                </span>
              );
            })}
          </div>
          <div
            style={{
              fontSize: '16px',
              color: 'gray',
              marginTop: '5px',
            }}
          >
            {ipaWord}
          </div>
        </div>
      );
    });
  };

  return (
    <div className="chat-root">
      <div
        className="chat-container shadow"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <h1 style={{ textAlign: 'center', padding: '20px' }}>
          {/* App Nhận Diện Phát Âm */}
        </h1>

        <div style={{ padding: '16px' }}>
          {displayText && (
            <div
              style={{
                marginBottom: '20px',
                fontSize: '18px',
                display: 'flex',
                justifyContent: 'center',
              }}
            >
              <div
                style={{
                  marginTop: '10px',
                  fontSize: '32px',
                  fontWeight: '600',
                }}
              >
                {renderText()}
              </div>
            </div>
          )}

          {displayText && (
            <div
              style={{
                display: 'flex',
                gap: '20px',
                justifyContent: 'center',
                marginTop: '20px',
              }}
            >
              <button
                onClick={handleMic}
                style={{
                  padding: '15px',
                  borderRadius: '50%',
                  backgroundColor: isMicActive ? '#a2d5ab' : '#f4a7a7',
                  color: '#0288d1',
                  border: 'none',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                  cursor: 'pointer',
                }}
              >
                🎤
              </button>

              <button
                onClick={handlePlayRecording}
                style={{
                  padding: '15px',
                  borderRadius: '50%',
                  backgroundColor: '#f1f8e9',
                  color: '#388e3c',
                  border: 'none',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                  cursor: 'pointer',
                }}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
            </div>
          )}

          {score ? (
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                marginTop: '20px',
              }}
            >
              <div
                style={{
                  fontSize: '24px',
                  fontWeight: '600',
                  color: score >= 50 ? 'green' : 'red',
                }}
              >
                {score.toFixed(1)}
              </div>
            </div>
          ) : null}

          {/* {ipaText && (
            <div style={{ marginBottom: '20px', fontSize: '18px' }}>
              <strong>IPA Phân Tích:</strong>
              <div
                style={{
                  marginTop: '10px',
                  display: 'flex',
                  flexWrap: 'wrap',
                }}
              >
                {ipaText}
              </div>
            </div>
          )}

          {predictedText && (
            <div style={{ marginBottom: '20px', fontSize: '18px' }}>
              <div style={{ color: 'blue', marginTop: '10px' }}>
                {predictedText}
              </div>
            </div>
          )}

          {matchingResult && (
            <div style={{ marginBottom: '20px', fontSize: '18px' }}>
              <div
                style={{
                  marginTop: '10px',
                  display: 'flex',
                  flexWrap: 'wrap',
                }}
              >
                {renderColoredText()}
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: '10px' }}>
            {ipaText && (
              <button
                onClick={handleMic}
                style={{ padding: '10px 15px', fontSize: '16px' }}
              >
                {isMicActive ? 'Tắt Mic' : 'Bật Mic'}
              </button>
            )}

            {recordedAudio && (
              <button
                onClick={handlePlayRecording}
                style={{
                  padding: '15px',
                  borderRadius: '50%',
                  backgroundColor: '#f1f8e9',
                  color: '#388e3c',
                  border: 'none',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                  cursor: 'pointer',
                }}
              >
                ▶
              </button>
            )}

            {generatedAudio && (
              <button
                onClick={handlePlayGeneratedAudio}
                style={{
                  padding: '15px',
                  borderRadius: '50%',
                  backgroundColor: '#00ff00',
                  color: '#388e3c',
                  border: 'none',
                  boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                  cursor: 'pointer',
                }}
              >
                Phát Âm Thanh Mẫu
              </button>
            )} 
          </div> */}
        </div>

        <div className="chat-input">
          <input
            type="text"
            value={inputText}
            onChange={handleInputChange}
            placeholder="Enter sentence to pronounce..."
          />
          <button
            onClick={handleConfirm}
            style={{ padding: '10px 15px', fontSize: '16px' }}
          >
            Submit
          </button>
        </div>

        <audio ref={audioPlayerRef} style={{ display: 'none' }} />
      </div>
    </div>
  );
}

export default PronunciationApp;
