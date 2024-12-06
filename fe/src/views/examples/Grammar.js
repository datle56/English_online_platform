import React, { useState } from 'react';
import './chat.css';

const GrammarForm = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [color, setColor] = useState(null);
  const [correctedSentence, setCorrectedSentence] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [mixSentence, setMixSentence] = useState(null);
  const [loading, setLoading] = useState(false);

  // Function to apply the colors based on the given color codes
  const applyColors = (text) => {
    const colorMapping = {
      1: 'green',
      2: '#888',
      3: '#4062f7',
    };

    let formattedText = '';
    let index = 0;

    while (index < text.length) {
      const currentChar = text[index];
      const colorCode = color.charAt(index);

      if (colorCode === '2') {
        formattedText += `<span style="text-decoration: line-through; text-decoration-color: red; text-decoration-thickness: 1px; color: ${colorMapping[colorCode]}">${currentChar}</span>`;
      } else if (colorMapping[colorCode]) {
        formattedText += `<span style="color: ${colorMapping[colorCode]}">${currentChar}</span>`;
      } else {
        formattedText += currentChar;
      }

      index++;
    }

    return formattedText;
  };

  const sendMessage = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');

      if (!token) {
        console.error('No authentication token found');
        return;
      }

      const response = await fetch('http://localhost:8000/grammar/correct', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ text: inputValue }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();

      // Assuming the backend returns both explanation and corrected_sentence
      setExplanation(data.explanation || 'No explanation provided');
      setCorrectedSentence(
        data.corrected_sentence || 'No corrected sentence available'
      );
      setMixSentence(data.mix_sentence || 'No mix sentence available');
      setColor(data.colour || '');
    } catch (error) {
      setExplanation(`Error: ${error.message}`);
      setCorrectedSentence(null); // Reset corrected sentence in case of error
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-root">
      <div className="chat-container shadow">
        <div className="messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}

          {explanation ? (
            <>
              <div>{explanation}</div>

              {correctedSentence && (
                <div
                  style={{
                    marginTop: '20px',
                    fontWeight: 'bold',
                    color: 'black',
                  }}
                >
                  <div
                    style={{
                      fontStyle: 'italic',
                    }}
                  >
                    {correctedSentence}
                  </div>
                </div>
              )}

              {mixSentence && (
                <div
                  style={{
                    marginTop: '10px',
                    fontWeight: 'bold',
                    color: 'black',
                  }}
                >
                  <div
                    style={{
                      fontStyle: 'italic',
                    }}
                    dangerouslySetInnerHTML={{
                      __html: applyColors(mixSentence),
                    }}
                  />
                </div>
              )}
            </>
          ) : (
            <div style={{ color: 'gray' }}>Response will appear here</div>
          )}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message..."
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button onClick={sendMessage}>
            {inputValue && loading ? 'Submitting' : 'Submit'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default GrammarForm;
