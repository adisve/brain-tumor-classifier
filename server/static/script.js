async function getPrediction() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    const formData = new FormData();
    
    formData.append('file', file);

    const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    alert('Predicted Label: ' + data.label);
}

function updateImagePreview() {
    console.log('Updating image preview')
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const imagePlaceholder = document.getElementById('image-placeholder');
        imagePlaceholder.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}