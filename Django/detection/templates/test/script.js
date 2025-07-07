// Update the selected option text based on menu clicks
document.querySelectorAll('nav ul li a').forEach(link => {
    link.addEventListener('click', (event) => {
        event.preventDefault();
        const option = link.textContent;
        document.getElementById('selected-option').textContent = `Upload ${option}`;
    });
});

// Handle file upload
document.getElementById('uploadBtn').addEventListener('click', () => {
    document.getElementById('fileUpload').click();
});

document.getElementById('fileUpload').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('statusText').textContent = 'Processing...';
        // Simulate processing delay
        setTimeout(() => {
            document.getElementById('statusText').textContent = 'Preview';
            const confidence = Math.floor(Math.random() * 100);
            document.getElementById('confidenceValue').textContent = `${confidence}%`;
        }, 2000);
    }
});