/**
 * TEAM-33 Frontend - Main Application Logic
 * Handles API communication, UI interactions, and state management
 */

// ==================== CONFIGURATION ====================

const API_BASE_URL = 'http://localhost:8000/api/v1';
const API_TIMEOUT = 30000; // 30 seconds
const BACKEND_URL = 'http://localhost:8000';

// ==================== STATE MANAGEMENT ====================

const state = {
    isConnected: false,
    currentMode: 'transcribe',
    isRecording: false,
    conversationHistory: [],
    sourceLanguage: 'en',
    targetLanguage: 'hi',
    enableTTS: true,
};

// ==================== DOM ELEMENTS ====================

const elements = {
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toastContainer: document.getElementById('toastContainer'),
    historyModal: document.getElementById('historyModal'),
    historyContent: document.getElementById('historyContent'),
};

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    checkServerStatus();
    loadSettings();
    setupDragAndDrop();
    console.log('✅ TEAM-33 Frontend initialized');
}

// ==================== EVENT LISTENERS ====================

function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', handleModeChange);
    });

    // File inputs
    const audioInputs = document.querySelectorAll('input[type="file"]');
    audioInputs.forEach((input, index) => {
        input.addEventListener('change', (e) => handleAudioUpload(e, index));
    });

    // Settings
    const sourceLanguage = document.getElementById('sourceLanguage');
    const targetLanguage = document.getElementById('targetLanguage');
    const enableTTS = document.getElementById('enableTTS');

    if (sourceLanguage) {
        sourceLanguage.addEventListener('change', (e) => {
            state.sourceLanguage = e.target.value;
            saveSettings();
        });
    }

    if (targetLanguage) {
        targetLanguage.addEventListener('change', (e) => {
            state.targetLanguage = e.target.value;
            saveSettings();
        });
    }

    if (enableTTS) {
        enableTTS.addEventListener('change', (e) => {
            state.enableTTS = e.target.checked;
            saveSettings();
        });
    }

    // Buttons
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const viewHistoryBtn = document.getElementById('viewHistoryBtn');
    const translateBtn = document.getElementById('translateBtn');

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearConversationHistory);
    }
    if (viewHistoryBtn) {
        viewHistoryBtn.addEventListener('click', showConversationHistory);
    }
    if (translateBtn) {
        translateBtn.addEventListener('click', handleTranslate);
    }

    // Voice button for voice translator
    const voiceBtn = document.getElementById('voiceBtn');
    if (voiceBtn) {
        voiceBtn.addEventListener('click', toggleVoiceRecording);
    }
}

function setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.upload-area');

    uploadAreas.forEach(area => {
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('drag-over');
        });

        area.addEventListener('dragleave', () => {
            area.classList.remove('drag-over');
        });

        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const audioFile = files[0];
                if (audioFile.type.startsWith('audio/')) {
                    const fileInput = area.querySelector('input[type="file"]');
                    if (fileInput) {
                        fileInput.files = files;
                        const event = { target: fileInput };
                        handleAudioUpload(event);
                    }
                } else {
                    showToast('Please upload an audio file', 'error');
                }
            }
        });
    });
}

// ==================== MODE NAVIGATION ====================

function handleModeChange(e) {
    e.preventDefault();
    const mode = e.target.dataset.mode;

    // Update active link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    e.target.classList.add('active');

    // Update active panel
    document.querySelectorAll('.mode-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    const panelId = `${mode}Panel`;
    const panel = document.getElementById(panelId);
    if (panel) {
        panel.classList.add('active');
    }

    state.currentMode = mode;
}

// ==================== AUDIO UPLOAD HANDLERS ====================

async function handleAudioUpload(event, index = 0) {
    const file = event.target.files[0];
    if (!file) return;

    showLoading('Processing audio...');

    try {
        let endpoint = '';
        let formData = new FormData();
        formData.append('file', file);

        // Determine which endpoint based on input type
        if (state.currentMode === 'transcribe') {
            endpoint = '/transcribe';
        } else if (state.currentMode === 'process') {
            endpoint = '/process';
            formData.append('target_language', state.targetLanguage);
            formData.append('with_tts', state.enableTTS);
        } else if (state.currentMode === 'healthcare') {
            endpoint = '/healthcare';
            formData.append('patient_name', 'Patient');
        }

        const response = await fetchWithTimeout(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (response.success) {
            if (endpoint === '/transcribe') {
                displayTranscriptionResult(response);
            } else if (endpoint === '/process') {
                displayProcessResult(response);
            } else if (endpoint === '/healthcare') {
                displayHealthcareResult(response);
            }
            showToast('Processing successful', 'success');
        } else {
            showToast(`Error: ${response.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        console.error('Upload error:', error);
    } finally {
        hideLoading();
    }
}

// ==================== RESULT DISPLAY ====================

function displayTranscriptionResult(data) {
    const resultDiv = document.getElementById('transcribeResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="result-card">
                <h3>📝 Transcription Result</h3>
                <p><strong>Text:</strong> ${data.transcription}</p>
                <p><strong>Language:</strong> ${data.language}</p>
                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                <p><strong>Model:</strong> ${data.model}</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    }
}

function displayProcessResult(data) {
    const resultDiv = document.getElementById('processResult');
    if (resultDiv) {
        let html = `
            <div class="result-card">
                <h3>🔄 Full Pipeline Result</h3>
                <p><strong>Original:</strong> ${data.transcription}</p>
                <p><strong>Translated:</strong> ${data.response}</p>
                <p><strong>Target Language:</strong> ${data.target_language}</p>
        `;

        if (data.audio_url) {
            html += `
                <div>
                    <strong>Audio:</strong>
                    <audio controls style="width: 100%; margin-top: 10px;">
                        <source src="${BACKEND_URL}${data.audio_url}" type="audio/mpeg">
                    </audio>
                </div>
            `;
        }

        html += `</div>`;
        resultDiv.innerHTML = html;
        resultDiv.style.display = 'block';
    }
}

function displayHealthcareResult(data) {
    const resultDiv = document.getElementById('healthcareResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="result-card">
                <h3>🏥 Healthcare Assessment</h3>
                <p><strong>Patient:</strong> ${data.patient}</p>
                <p><strong>Query:</strong> ${data.query}</p>
                <p><strong>Response:</strong> ${data.response}</p>
                <p style="color: #ff6b6b; font-weight: bold;">⚠️ ${data.disclaimer}</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    }
}

// ==================== TRANSLATION ====================

async function handleTranslate() {
    const text = document.getElementById('translateInput')?.value.trim();
    if (!text) {
        showToast('Please enter text to translate', 'warning');
        return;
    }

    showLoading('Translating...');

    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('source_language', state.sourceLanguage);
        formData.append('target_language', state.targetLanguage);

        const response = await fetchWithTimeout(`${API_BASE_URL}/translate`, {
            method: 'POST',
            body: formData,
        });

        if (response.success) {
            const resultDiv = document.getElementById('translateResult');
            if (resultDiv) {
                resultDiv.innerHTML = `
                    <div class="result-card">
                        <h3>🌐 Translation Result</h3>
                        <p><strong>Original (${response.source_language}):</strong></p>
                        <p>${response.original_text}</p>
                        <p><strong>Translation (${response.target_language}):</strong></p>
                        <p>${response.response}</p>
                    </div>
                `;
                resultDiv.style.display = 'block';
            }
            showToast('Translation complete', 'success');
        } else {
            showToast(`Error: ${response.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// ==================== VOICE RECORDING ====================

let recognition = null;
let synthesizer = null;

function initializeVoiceAPIs() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showToast('Speech Recognition not supported in this browser', 'error');
        return false;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.language = 'en-US';

    recognition.onstart = () => {
        state.isRecording = true;
        updateVoiceButton();
    };

    recognition.onresult = handleVoiceResult;
    recognition.onerror = handleVoiceError;
    recognition.onend = () => {
        state.isRecording = false;
        updateVoiceButton();
    };

    return true;
}

function toggleVoiceRecording() {
    if (!recognition) {
        if (!initializeVoiceAPIs()) return;
    }

    if (state.isRecording) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

function handleVoiceResult(event) {
    let text = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
        text += event.results[i][0].transcript;
    }

    if (event.isFinal) {
        processVoiceTranslation(text);
    }
}

function handleVoiceError(event) {
    showToast(`Error: ${event.error}`, 'error');
}

async function processVoiceTranslation(text) {
    showLoading('Translating...');

    try {
        const targetLang = document.getElementById('targetLanguage')?.value || 'hi';
        const formData = new FormData();
        formData.append('text', text);
        formData.append('source_language', 'en');
        formData.append('target_language', targetLang);

        const response = await fetchWithTimeout(`${API_BASE_URL}/translate`, {
            method: 'POST',
            body: formData,
        });

        if (response.success) {
            displayVoiceTranslation(text, response.response);
            if (state.enableTTS) {
                speakTranslation(response.response, targetLang);
            }
        }
    } catch (error) {
        showToast(`Translation error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function displayVoiceTranslation(original, translated) {
    const resultsContainer = document.getElementById('voiceResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="voice-results">
                <div class="result-panel">
                    <h4>🎤 You said (English)</h4>
                    <p>${original}</p>
                </div>
                <div class="result-panel">
                    <h4>🌐 Translation</h4>
                    <p>${translated}</p>
                </div>
            </div>
        `;
        resultsContainer.style.display = 'block';
    }
}

function speakTranslation(text, language) {
    if (!('speechSynthesis' in window)) {
        showToast('Speech Synthesis not supported', 'warning');
        return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = `${language}-${language.toUpperCase()}`;
    speechSynthesis.speak(utterance);
}

function updateVoiceButton() {
    const voiceBtn = document.getElementById('voiceBtn');
    if (voiceBtn) {
        if (state.isRecording) {
            voiceBtn.textContent = '⏹️ STOP LISTENING';
            voiceBtn.classList.add('recording');
        } else {
            voiceBtn.textContent = '🎙️ START LISTENING';
            voiceBtn.classList.remove('recording');
        }
    }
}

// ==================== CONVERSATION HISTORY ====================

async function clearConversationHistory() {
    if (!confirm('Clear all conversation history?')) return;

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/conversation`, {
            method: 'DELETE',
        });

        if (response.success) {
            state.conversationHistory = [];
            showToast('Conversation cleared', 'success');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

async function showConversationHistory() {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/conversation`, {
            method: 'GET',
        });

        if (response.success && response.history) {
            displayConversationHistory(response.history);
            elements.historyModal.style.display = 'flex';
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

function displayConversationHistory(history) {
    if (history.length === 0) {
        elements.historyContent.innerHTML = '<p>No conversation history</p>';
        return;
    }

    const html = history
        .map(
            (item) => `
        <div class="history-item">
            <strong>${item.role === 'user' ? '👤 You' : '🤖 Assistant'}:</strong>
            <p>${item.content}</p>
            <small>${new Date(item.timestamp).toLocaleString()}</small>
        </div>
    `
        )
        .join('');

    elements.historyContent.innerHTML = html;
}

function closeModal() {
    elements.historyModal.style.display = 'none';
}

// ==================== SERVER STATUS ====================

async function checkServerStatus() {
    try {
        const response = await fetchWithTimeout(`${BACKEND_URL}/health`, {
            method: 'GET',
        });

        if (response.status === 'healthy') {
            updateServerStatus(true);
        }
    } catch (error) {
        updateServerStatus(false);
        console.warn('⚠️ Backend server unreachable');
    }
}

function updateServerStatus(isConnected) {
    state.isConnected = isConnected;

    if (elements.statusDot && elements.statusText) {
        if (isConnected) {
            elements.statusDot.classList.remove('offline');
            elements.statusDot.classList.add('online');
            elements.statusText.textContent = '🟢 Connected';
            elements.statusText.style.color = '#4CAF50';
        } else {
            elements.statusDot.classList.add('offline');
            elements.statusDot.classList.remove('online');
            elements.statusText.textContent = '🔴 Offline';
            elements.statusText.style.color = '#ff6b6b';
        }
    }
}

// ==================== UI HELPERS ====================

function showLoading(text = 'Processing...') {
    if (elements.loadingText) {
        elements.loadingText.textContent = text;
    }
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = 'flex';
    }
}

function hideLoading() {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    if (elements.toastContainer) {
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// ==================== SETTINGS ====================

function saveSettings() {
    localStorage.setItem(
        'team33-settings',
        JSON.stringify({
            sourceLanguage: state.sourceLanguage,
            targetLanguage: state.targetLanguage,
            enableTTS: state.enableTTS,
        })
    );
}

function loadSettings() {
    const settings = localStorage.getItem('team33-settings');
    if (settings) {
        const parsed = JSON.parse(settings);
        state.sourceLanguage = parsed.sourceLanguage || 'en';
        state.targetLanguage = parsed.targetLanguage || 'hi';
        state.enableTTS = parsed.enableTTS !== false;

        // Update UI
        const sourceSelect = document.getElementById('sourceLanguage');
        const targetSelect = document.getElementById('targetLanguage');
        const ttsCheck = document.getElementById('enableTTS');

        if (sourceSelect) sourceSelect.value = state.sourceLanguage;
        if (targetSelect) targetSelect.value = state.targetLanguage;
        if (ttsCheck) ttsCheck.checked = state.enableTTS;
    }
}

// ==================== UTILITY FUNCTIONS ====================

async function fetchWithTimeout(url, options = {}) {
    const timeout = options.timeout || API_TIMEOUT;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
            throw new Error('Request timeout');
        }

        throw error;
    }
}

// ==================== EVENT LISTENERS ====================

window.addEventListener('click', (event) => {
    if (event.target === elements.historyModal) {
        closeModal();
    }
});

// Auto-check server status every 10 seconds
setInterval(checkServerStatus, 10000);

// Log initialization
console.log('%c✅ TEAM-33 Frontend v2.0.0', 'color: #00ff00; font-size: 14px; font-weight: bold;');
console.log(`API Base URL: ${API_BASE_URL}`);
console.log(`Backend URL: ${BACKEND_URL}`);
