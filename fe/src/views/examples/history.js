import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Table from 'react-bootstrap/Table';
import Button from 'react-bootstrap/Button';
import './chat.css';

import historyData from '../../data/history.json';

function History() {
  const [history, setHistory] = useState(historyData); // Store all history data
  const [filteredHistory, setFilteredHistory] = useState(historyData); // Store filtered history
  const [selectedFeature, setSelectedFeature] = useState(''); // Store selected feature

  // Fetch the user's history data
  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:8000/history', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setHistory(response.data); // Update all history data
    } catch (error) {
      console.error('Error fetching history', error);
    }
  };

  // Filter and sort the history data based on selected feature and date
  const handleFeatureClick = (feature) => {
    setSelectedFeature(feature); // Set the selected feature

    // Filter the history by feature and sort by created_at (most recent first)
    const filtered = history
      .filter((item) => item.feature === feature)
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    setFilteredHistory(filtered); // Update filtered history
  };

  useEffect(() => {
    fetchHistory(); // Fetch history data when the component is mounted
  }, []);

  // Filter the history data whenever `history` or `selectedFeature` changes
  useEffect(() => {
    const filtered = history
      .filter((item) =>
        selectedFeature ? item.feature === selectedFeature : true
      )
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    setFilteredHistory(filtered); // Update filtered history
  }, [history, selectedFeature]);

  const coloredText = (inputData, outputData) => {
    return inputData.split('').map((char, index) => {
      const color =
        outputData.charAt(index) === '0'
          ? 'red'
          : outputData.charAt(index) === '1'
          ? 'green'
          : 'black';

      return (
        <span key={index} style={{ color: color }}>
          {char}
        </span>
      );
    });
  };

  return (
    <div className="chat-root">
      <div
        className="chat-container shadow"
        style={{ padding: '20px', overflowY: 'scroll' }}
      >
        <h2>History</h2>

        {/* Feature Buttons */}
        <div className="mb-3">
          <Button
            variant="primary"
            onClick={() => handleFeatureClick('Pronunciation')}
            active={selectedFeature === 'Pronunciation'}
          >
            Pronunciation
          </Button>
          <Button
            variant="secondary"
            onClick={() => handleFeatureClick('Grammar')}
            active={selectedFeature === 'Grammar'}
            className="ml-2"
          >
            Grammar
          </Button>
          <Button
            variant="success"
            onClick={() => handleFeatureClick('Talk')}
            active={selectedFeature === 'Talk'}
            className="ml-2"
          >
            Talk
          </Button>
        </div>

        {/* History Table */}
        <Table striped bordered hover>
          <thead>
            <tr>
              <th>ID</th>
              <th>Feature</th>
              <th>Input Data</th>
              <th>Output Data</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {filteredHistory.map((historyItem, index) => (
              <tr key={index}>
                <td
                  style={{
                    wordWrap: 'break-word',
                    wordBreak: 'break-word',
                    maxWidth: '200px',
                    whiteSpace: 'normal',
                  }}
                >
                  {historyItem.id}
                </td>
                <td
                  style={{
                    wordWrap: 'break-word',
                    wordBreak: 'break-word',
                    maxWidth: '200px',
                    whiteSpace: 'normal',
                  }}
                >
                  {historyItem.feature}
                </td>
                <td
                  style={{
                    wordWrap: 'break-word',
                    wordBreak: 'break-word',
                    maxWidth: '200px',
                    whiteSpace: 'normal',
                  }}
                >
                  {historyItem.input_data}
                </td>
                <td
                  style={{
                    wordWrap: 'break-word',
                    wordBreak: 'break-word',
                    maxWidth: '200px',
                    whiteSpace: 'normal',
                  }}
                >
                  {historyItem.feature === 'Pronunciation'
                    ? coloredText(
                        historyItem.input_data,
                        historyItem.output_data
                      )
                    : historyItem.output_data}
                </td>
                <td
                  style={{
                    wordWrap: 'break-word',
                    wordBreak: 'break-word',
                    maxWidth: '200px',
                    whiteSpace: 'normal',
                  }}
                >
                  {new Date(historyItem.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    </div>
  );
}

export default History;
