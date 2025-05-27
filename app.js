import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js';
import { getAuth, GoogleAuthProvider, signInWithPopup, onAuthStateChanged, getIdToken, signOut } from 'https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js';

const firebaseConfig = {
    apiKey: "AIzaSyDCJ5RG-hfpdC1uNc44T1KaLrELM1Mud7o",
    authDomain: "rag-user-management.firebaseapp.com",
    projectId: "rag-user-management",
    storageBucket: "rag-user-management.firebasestorage.app",
    messagingSenderId: "741967144538",
    appId: "1:741967144538:web:8ca101a2c13f14ef141a63",
    measurementId: "G-KXMMMX2YTZ"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const BACKEND_URL = 'http://localhost:8000';

// Index Page
const signInBtn = document.getElementById('signInBtn');
if (signInBtn) {
    const provider = new GoogleAuthProvider();
    onAuthStateChanged(auth, user => {
        if (user) {
            window.location.href = 'chat.html';
        }
    });
    signInBtn.addEventListener('click', async () => {
        try {
            await signInWithPopup(auth, provider);
        } catch (error) {
            console.error('Sign-in error:', error);
            document.getElementById('error').textContent = `Failed to sign in: ${error.message}`;
        }
    });
}

// Chat Page
const chatElements = {
    userName: document.getElementById('userName'),
    status: document.getElementById('status'),
    manageDocsBtn: document.getElementById('manageDocsBtn'),
    signOutBtn: document.getElementById('signOutBtn'),
    response: document.getElementById('response'),
    questionInput: document.getElementById('questionInput'),
    sendBtn: document.getElementById('sendBtn')
};

if (chatElements.userName) {
    onAuthStateChanged(auth, async user => {
        if (!user) {
            window.location.href = 'index.html';
        } else {
            chatElements.userName.textContent = user.displayName || 'User';
            updateStatus();
        }
    });

    async function fetchIdToken() {
        return await getIdToken(auth.currentUser);
    }

    async function updateStatus() {
        try {
            const token = await fetchIdToken();
            const statusResponse = await fetch(`${BACKEND_URL}/status`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!statusResponse.ok) throw new Error(`Status request failed: ${statusResponse.status}`);
            const statusData = await statusResponse.json();
            const docCount = statusData.documents_count;
            let statusMessage = 'No documents uploaded';
            if (docCount > 0) {
                const chunkCount = await getChunkCount();
                statusMessage = `System ready with ${docCount} document${docCount !== 1 ? 's' : ''} (${chunkCount} chunk${chunkCount !== 1 ? 's' : ''})`;
            }
            chatElements.status.textContent = statusMessage;
        } catch (error) {
            console.error('Error updating status:', error);
            chatElements.status.textContent = 'Error fetching status';
        }
    }

    async function getChunkCount() {
        try {
            const token = await fetchIdToken();
            const response = await fetch(`${BACKEND_URL}/documents`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!response.ok) throw new Error(`Documents request failed: ${response.status}`);
            const data = await response.json();
            return data.documents.reduce((total, doc) => total + doc.chunks, 0);
        } catch (error) {
            console.error('Error fetching chunk count:', error);
            return 0;
        }
    }

    async function askQuestion() {
        const question = chatElements.questionInput.value;
        if (!question) return;
        try {
            const token = await fetchIdToken();
            const response = await fetch(`${BACKEND_URL}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ question })
            });
            const data = await response.json();
            chatElements.response.innerHTML = `
                <p><strong>Answer:</strong> ${data.answer}</p>
                <p><strong>Sources:</strong> ${data.sources.join(', ')}</p>
                <p><strong>LLM Used:</strong> ${data.llm_used}</p>
            `;
            chatElements.questionInput.value = '';
        } catch (error) {
            console.error('Error asking question:', error);
            chatElements.response.innerHTML = `<p>Error: ${error.message}</p>`;
        }
    }

    chatElements.sendBtn?.addEventListener('click', askQuestion);
    chatElements.manageDocsBtn?.addEventListener('click', () => {
        window.location.href = 'documents.html';
    });
    chatElements.signOutBtn?.addEventListener('click', async () => {
        try {
            await signOut(auth);
            window.location.href = 'index.html';
        } catch (error) {
            console.error('Sign-out failed:', error);
            alert('Failed to sign out: ' + error.message);
        }
    });
}

// Documents Page
const docElements = {
    userName: document.getElementById('userName'),
    backToChatBtn: document.getElementById('backToChatBtn'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    docList: document.getElementById('docList'),
    clearBtn: document.getElementById('clearBtn')
};

if (docElements.userName) {
    onAuthStateChanged(auth, async user => {
        if (!user) {
            window.location.href = 'index.html';
        } else {
            docElements.userName.textContent = user.displayName || 'User';
            listDocuments();
        }
    });

    async function fetchIdToken() {
        return await getIdToken(auth.currentUser);
    }

    async function listDocuments() {
        try {
            const token = await fetchIdToken();
            const response = await fetch(`${BACKEND_URL}/documents`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!response.ok) throw new Error(`Documents request failed: ${response.status}`);
            const data = await response.json();
            docElements.docList.innerHTML = '';
            if (data.documents.length === 0) {
                docElements.docList.innerHTML = '<li>No documents uploaded yet</li>';
            } else {
                data.documents.forEach(doc => {
                    const li = document.createElement('li');
                    li.textContent = `${doc.source} (${doc.chunks} chunks, Processed: ${doc.processed ? 'Yes' : 'No'})`;
                    docElements.docList.appendChild(li);
                });
            }
        } catch (error) {
            console.error('Error listing documents:', error);
            docElements.docList.innerHTML = '<li>Error fetching documents</li>';
        }
    }

    async function uploadFiles() {
        const files = docElements.fileInput.files;
        if (files.length === 0) return;
        try {
            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }
            const token = await fetchIdToken();
            await fetch(`${BACKEND_URL}/upload`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });
            listDocuments();
        } catch (error) {
            console.error('Error uploading files:', error);
            alert('Failed to upload files: ' + error.message);
        }
    }

    async function clearDocuments() {
        try {
            const token = await fetchIdToken();
            await fetch(`${BACKEND_URL}/documents`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            listDocuments();
        } catch (error) {
            console.error('Error clearing documents:', error);
            alert('Failed to clear documents: ' + error.message);
        }
    }

    docElements.uploadBtn?.addEventListener('click', uploadFiles);
    docElements.clearBtn?.addEventListener('click', clearDocuments);
    docElements.backToChatBtn?.addEventListener('click', () => {
        window.location.href = 'chat.html';
    });
}