import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Table from 'react-bootstrap/Table';
import Button from 'react-bootstrap/Button';

function UserList() {
    const [users, setUsers] = useState([]);

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const token = localStorage.getItem('token');
                if (!token) {
                    throw new Error("Token is missing. Please log in again.");
                }
                const response = await axios.get('http://127.0.0.1:8000/admin/users', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                setUsers(response.data);
            } catch (error) {
                if (error.response && error.response.status === 401) {
                    alert("Unauthorized: Please log in again.");
                } else {
                    console.error("Error fetching users:", error);
                }
            }
        };
        

        fetchUsers();
    }, []);

    const deleteUser = async (id) => {
        const confirmDelete = window.confirm("Are you sure you want to delete this user?");
        if (confirmDelete) {
            try {
                const token = localStorage.getItem('token');
                await axios.delete(`https://speak.id.vn/api/admin/users/${id}`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                setUsers(users.filter(user => user.id !== id));
            } catch (error) {
                console.error("Error deleting user:", error);
            }
        }
    };

    return (
        <div>
            <h1>Users</h1>
            <Table striped bordered hover>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Username</th>
                        <th>Email</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {users.map(user => (
                        <tr key={user.id}>
                            <td>{user.id}</td>
                            <td>{user.username}</td>
                            <td>{user.email}</td>
                            <td>
                                <Button variant="danger" onClick={() => deleteUser(user.id)}>Delete</Button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </Table>
        </div>
    );
}

export default UserList;
