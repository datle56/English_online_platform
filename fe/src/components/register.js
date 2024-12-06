import React, { useState } from 'react';
import { Button, Col, Row, Input, Space, message, Image } from 'antd';
import {
  HomeOutlined,
  EyeInvisibleOutlined,
  EyeTwoTone,
} from '@ant-design/icons';
import { Link, NavLink, useNavigate } from 'react-router-dom';

const linkStyle = {
  textDecoration: 'underline',
  color: '#bdc3c7',
};

function Register() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const navigate = useNavigate();

  const handleSubmit = async () => {
    if (password !== confirmPassword) {
      message.error('Passwords do not match');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:8000/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username,
          email,
          password,
          role: 'user', // Default role set to 'user'
        }),
      });

      if (response.ok) {
        message.success('Registration successful');
        navigate('/login');
      } else {
        const data = await response.json();
        if (data.detail === 'Username already exists') {
          message.error('Username already exists');
        } else {
          message.error('Registration failed');
        }
      }
    } catch (error) {
      console.error('Error during registration:', error);
      message.error('An error occurred during registration');
    }
  };

  return (
    <div>
      <Row>
        <Col
          span={12}
          style={{
            display: 'flex',
            flexDirection: 'column',
            padding: '4rem',
          }}
        >
          <Link
            to="/"
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
          </Link>
          <div className="title" style={{ marginTop: '2rem' }}>
            <p style={{ fontSize: '2.3rem', color: 'black', fontWeight: 500 }}>
              Sign up to <span style={{ fontWeight: 700 }}>Speak</span>
            </p>
            <p>Welcome to SPEAK</p>
          </div>
          <div className="email-password-input">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                style={{ padding: '8px 20px', borderRadius: '12px' }}
              />
              <Input
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{
                  marginTop: '4px',
                  padding: '8px 20px',
                  borderRadius: '12px',
                }}
              />
              <Input.Password
                placeholder="Password"
                iconRender={(visible) =>
                  visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />
                }
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{
                  marginTop: '4px',
                  padding: '8px 20px',
                  borderRadius: '12px',
                }}
              />
              <Input.Password
                placeholder="Confirm password"
                iconRender={(visible) =>
                  visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />
                }
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                style={{
                  marginTop: '4px',
                  padding: '8px 20px',
                  borderRadius: '12px',
                }}
              />
            </Space>
          </div>

          <button
            style={{
              width: '100%',
              backgroundColor: 'black',
              color: 'white',
              padding: '10px 24px',
              borderRadius: '12px',
              cursor: 'pointer',
              textDecoration: 'none',
              fontSize: '14px',
              fontWeight: '500',
              marginTop: '24px',
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
            onClick={handleSubmit}
          >
            Register
          </button>

          <div
            className="link-sign-tup-login-tutor"
            style={{ marginTop: '2rem' }}
          >
            <Link to="/login" style={{ ...linkStyle, marginRight: '4rem' }}>
              Or Login
            </Link>
          </div>
        </Col>
        <Col span={12} style={{ padding: '6rem', width: '110%' }}>
          <Image src="https://www.shutterstock.com/image-vector/man-key-near-computer-account-260nw-1499141258.jpg" />
        </Col>
      </Row>

      <div
        className="footer1"
        style={{
          width: '100%',
          backgroundColor: '#000',
          color: 'white',
          height: 'fit-content',
        }}
      >
        <Row>
          <Col span={6} style={{ padding: '2rem', marginTop: '30px' }}>
            <h3
              style={{
                color: '#fff',
                fontSize: 'large',
                fontWeight: 'bold',
                marginBottom: '18px',
              }}
            >
              ABOUT US
            </h3>
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Mission & Vision
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Our Company
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Our Projects
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Our Team
            </NavLink>
          </Col>
          <Col span={6} style={{ padding: '2rem', marginTop: '30px' }}>
            <h3
              style={{
                color: '#fff',
                fontSize: 'large',
                fontWeight: 'bold',
                marginBottom: '18px',
              }}
            >
              DISCOVER
            </h3>
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Projects & Research
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Clients Review
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Our Projects
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Our Team
            </NavLink>
          </Col>
          <Col span={6} style={{ padding: '2rem', marginTop: '30px' }}>
            <h3
              style={{
                color: '#fff',
                fontSize: 'large',
                fontWeight: 'bold',
                marginBottom: '18px',
              }}
            >
              USEFUL LINKS
            </h3>
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Contact Us
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Terms & Conditions
            </NavLink>
            <br />
            <NavLink
              style={{
                color: 'white',
                margin: '1rem 0px 0px ',
                fontSize: 'mediumm',
                fontWeight: '500',
              }}
              to="/"
            >
              Review
            </NavLink>
          </Col>
          <Col span={6} style={{ padding: '2rem', marginTop: '30px' }}>
            <h3
              style={{
                color: '#fff',
                fontSize: 'large',
                fontWeight: 'bold',
                marginBottom: '18px',
              }}
            >
              SPEAK
            </h3>
            <p style={{ color: '#B2B3CF' }}>
              Seize Potential, Enhance & Acquire Knowledge
            </p>
            <p style={{ color: '#B2B3CF' }}>Subscribe to get our Newsletter</p>
            <Input
              placeholder="Enter email"
              style={{
                background: 'transparent',
                borderRadius: '30px',
                width: '100%',
                marginBottom: '1rem',
                color: 'white',
              }}
            />
            <Button
              variant="primary"
              type="submit"
              style={{
                width: '100%',
                backgroundColor: 'white',
                color: 'black',
              }}
            >
              Submit
            </Button>
            <p style={{ color: 'white', marginTop: '1rem' }}>
              We'll never share your email with anyone else.
            </p>
          </Col>
        </Row>
        <Row>
          <Col
            span={24}
            style={{
              textAlign: 'center',
              color: '#B2B3CF',
              fontSize: '15px',
              marginBottom: '30px',
            }}
          >
            © 2021 Class Technologies Inc.
          </Col>
        </Row>
      </div>
    </div>
  );
}

export default Register;
