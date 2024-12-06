import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

function Homepage() {
  const [bgIndex, setBgIndex] = useState(0);

  const backgrounds = [
    'url("https://images.unsplash.com/photo-1565598621680-94ac0c22b148?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")',
    'url("https://images.unsplash.com/photo-1509062522246-3755977927d7?q=80&w=2132&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")',
    'url("https://images.unsplash.com/photo-1443188631128-a1b6b1c5c207?q=80&w=1931&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")',
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setBgIndex((prevIndex) => (prevIndex + 1) % backgrounds.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        width: '100%',
        minHeight: '100vh',
        background: 'white',
      }}
    >
      <nav
        style={{
          display: 'flex',
          maxWidth: '1240px',
          margin: '0 auto',
          justifyContent: 'space-between',
          width: '100%',
          height: '72px',
          padding: '12px 20px',
          alignItems: 'center',
        }}
      >
        <h1
          style={{
            fontWeight: 700,
            margin: 0,
            color: 'black',
            cursor: 'pointer',
            fontSize: '28px',
            letterSpacing: '1px',
          }}
        >
          Speak
        </h1>

        <div style={{ display: 'flex', gap: '16px' }}>
          <Link
            to="/login"
            style={{
              backgroundColor: 'white',
              color: 'black',
              padding: '8px 24px',
              borderRadius: '12px',
              cursor: 'pointer',
              border: '2px solid #888',
              textDecoration: 'none',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'background-color 0.3s ease, border-color 0.3s ease',
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#f0f0f0';
              e.target.style.borderColor = '#555';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'white';
              e.target.style.borderColor = '#888';
            }}
          >
            Login
          </Link>
          <Link
            to="/register"
            style={{
              backgroundColor: 'black',
              color: 'white',
              padding: '8px 24px',
              borderRadius: '12px',
              cursor: 'pointer',
              textDecoration: 'none',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'background-color 0.3s ease, transform 0.3s ease',
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#444';
              e.target.style.transform = 'scale(1.02)';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'black';
              e.target.style.transform = 'scale(1)';
            }}
          >
            Register
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          height: '80vh',
          backgroundColor: '#333',
          color: 'white',
          padding: '0 20px',
          backgroundImage: backgrounds[bgIndex],
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundBlendMode: 'overlay',
          backgroundColor: 'rgba(0, 0, 0, 0.2)',
        }}
      >
        <h2 style={{ fontSize: '64px', fontWeight: '700', color: 'white' }}>
          Welcome to Speak
        </h2>
        <p style={{ fontSize: '32px', fontWeight: 600, color: '#eee' }}>
          Enhance your language skills with AI-powered tools
        </p>
      </section>

      {/* Main Functionality Cards */}
      <section
        style={{
          maxWidth: '1240px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          margin: '0 auto',
          padding: '40px 0 80px 0',
        }}
      >
        <h3
          style={{
            fontSize: '28px',
            fontWeight: '700',
            color: '#333',
            marginBottom: '40px',

            marginBottom: '0px',
          }}
        >
          Explore Our Features
        </h3>

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            flexWrap: 'wrap',
          }}
        >
          <div
            style={{
              backgroundColor: '#fafafa',
              padding: '30px',
              borderRadius: '12px',
              boxShadow: '0 4px 10px rgba(0, 0, 0, 0.1)',
              width: '380px',

              marginTop: '20px',
            }}
          >
            <h3 style={{ fontSize: '20px', fontWeight: '600', color: 'black' }}>
              Grammar Check
            </h3>
            <p style={{ fontSize: '14px', color: '#555' }}>
              Improve your writing skills with the AI-powered smart grammar
              checker. It helps you detect and correct grammar errors in your
              writing, ensuring your writing is clearer and more accurate.
            </p>
          </div>

          <div
            style={{
              backgroundColor: '#fafafa',
              padding: '30px',
              borderRadius: '12px',
              boxShadow: '0 4px 10px rgba(0, 0, 0, 0.1)',
              width: '380px',

              marginTop: '20px',
            }}
          >
            <h3 style={{ fontSize: '20px', fontWeight: '600', color: 'black' }}>
              Pronunciation Check
            </h3>
            <p style={{ fontSize: '14px', color: '#555' }}>
              Improve your pronunciation with AI-powered tests and feedback. It
              helps you identify and correct pronunciation errors, giving you
              more confidence when communicating in real-life situations.
            </p>
          </div>

          <div
            style={{
              backgroundColor: '#fafafa',
              padding: '30px',
              borderRadius: '12px',
              boxShadow: '0 4px 10px rgba(0, 0, 0, 0.1)',
              width: '380px',
              marginTop: '20px',
            }}
          >
            <h3 style={{ fontSize: '20px', fontWeight: '600', color: 'black' }}>
              Message with AI
            </h3>
            <p style={{ fontSize: '14px', color: '#555' }}>
              Chat with AI to practice your language skills in a natural and fun
              way. You can practice conversations, improve your language
              reflexes, and expand your vocabulary through intelligent and
              lively conversations.
            </p>
          </div>
        </div>
      </section>

      <footer
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: '#333',
          color: 'white',
          fontSize: '14px',
          padding: '12px 0',
        }}
      >
        <p style={{ marginBottom: '0px', fontWeight: 500 }}>
          &copy; 2024 <span style={{ fontWeight: 600 }}>Speak</span>. All rights
          reserved.
        </p>
      </footer>
    </div>
  );
}

export default Homepage;
