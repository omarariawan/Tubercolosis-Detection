// Configuration
const API_URL = 'http://localhost:5000';
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const previewImage = document.getElementById('previewImage');
const previewContainer = document.getElementById('previewContainer');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsContent = document.getElementById('resultsContent');
const loadingSpinner = document.getElementById('loadingSpinner');

let selectedFile = null;

// ==================== FILE UPLOAD HANDLERS ====================

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// ==================== FILE HANDLING ====================

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file.');
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showError('File size exceeds 10MB limit.');
        return;
    }

    selectedFile = file;

    // Update file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'block';

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        previewContainer.innerHTML = '';
        previewContainer.appendChild(previewImage);
    };
    reader.readAsDataURL(file);

    // Enable predict button
    predictBtn.disabled = false;

    // Reset results
    resultsContent.innerHTML = '<p class="no-results">Ready to analyze. Click "Analyze Image" to start.</p>';
}

// ==================== PREDICTION ====================

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    await analyzeImage(selectedFile);
});

async function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Show loading
        showLoading(true);

        // Send request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server error: ${response.status}`);
        }

        if (data.status !== 'success') {
            throw new Error(data.error || 'Prediction failed');
        }

        // Display results
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        
        // Check if it's a connection error
        if (error.message.includes('Failed to fetch')) {
            showError('❌ Cannot connect to backend.\nMake sure the server is running on http://localhost:5000\nRun: python app.py');
        } else {
            showError(`Error: ${error.message}`);
        }
    } finally {
        showLoading(false);
    }
}

// ==================== RESULTS DISPLAY ====================

function displayResults(result) {
    if (result.status !== 'success') {
        showError('Prediction failed. Please try again.');
        return;
    }

    const prediction = result.prediction;
    const filename = result.filename;

    let resultHTML = `
        <div class="result-success">
            <h3>Analysis Complete</h3>
            <div class="result-item">
                <div class="result-label">File Name</div>
                <div class="result-value">${filename}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Prediction Result</div>
                <div class="result-value">
                    <span class="prediction-badge ${prediction.toLowerCase() === 'normal' ? 'normal' : 'tuberculosis'}">
                        ${prediction}
                    </span>
                </div>
            </div>
            <div class="result-item">
                <div class="result-label">Status</div>
                <div class="result-value" style="color: ${prediction.toLowerCase() === 'normal' ? 'var(--success-color)' : 'var(--danger-color)'};">
                    ${getPredictionStatus(prediction)}
                </div>
            </div>
    `;

    // Add confidence if available
    if (result.confidence) {
        const confidencePercent = (result.confidence * 100).toFixed(2);
        resultHTML += `
            <div class="result-item">
                <div class="result-label">Confidence Score</div>
                <div class="result-value">${confidencePercent}%</div>
            </div>
        `;
    }

    resultHTML += `
            <p style="margin-top: 20px; font-size: 0.9rem; color: var(--text-secondary);">
                <strong>Disclaimer:</strong> This analysis is for educational purposes. 
                Always consult with a medical professional for diagnosis.
            </p>
        </div>
    `;

    resultsContent.innerHTML = resultHTML;
}

function getPredictionStatus(prediction) {
    const lower = prediction.toLowerCase();
    if (lower === 'normal') {
        return '✓ Normal - No signs of tuberculosis detected';
    } else if (lower === 'tuberculosis') {
        return '⚠ Positive - TB indicators detected. Consult a doctor immediately.';
    }
    return 'Unknown';
}

// ==================== UI HELPERS ====================

function showLoading(isLoading) {
    if (isLoading) {
        loadingSpinner.style.display = 'flex';
        loadingSpinner.classList.add('active');
        predictBtn.disabled = true;
    } else {
        loadingSpinner.style.display = 'none';
        loadingSpinner.classList.remove('active');
        predictBtn.disabled = false;
    }
}

function showError(message) {
    resultsContent.innerHTML = `<div class="error-message">${message}</div>`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// ==================== CLEAR BUTTON ====================

clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    previewImage.style.display = 'none';
    previewContainer.innerHTML = '<p class="no-image">No image selected</p>';
    resultsContent.innerHTML = '<p class="no-results">No prediction yet. Please upload an X-Ray image.</p>';
    predictBtn.disabled = true;
});

// ==================== API HEALTH CHECK ====================

document.addEventListener('DOMContentLoaded', async () => {
    checkBackendConnection();
});

async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'OPTIONS'
        }).catch(() => {
            // OPTIONS may not be supported, try POST without file
            return fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
        });
        
        // If we get here, the server is running
        document.getElementById('apiStatus').textContent = `${API_URL} ✓ Connected`;
        document.getElementById('apiStatus').style.color = 'var(--success-color)';
        console.log('✓ Backend API is connected');
    } catch (error) {
        document.getElementById('apiStatus').textContent = `${API_URL} ✗ Not Connected`;
        document.getElementById('apiStatus').style.color = 'var(--danger-color)';
        console.warn('⚠ Backend API is not accessible. Make sure it\'s running on http://localhost:5000');
        showError('❌ Backend not connected. Please start the backend server:\nRun: python app.py');
    }
}

// ==================== KEYBOARD SHORTCUTS ====================

document.addEventListener('keydown', (e) => {
    // Ctrl+V or Cmd+V to paste image
    if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        // This would require clipboard API permissions
        // For now, just hint the user
    }
});
