async function getPrediction(model) {
    console.log(model);

    let predictionText = document.getElementById("prediction-text");
    predictionText.innerText = `Predicted Label: Loading...`;
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];
    const formData = new FormData();

    formData.append("file", file);

    const response = await fetch(
        `http://127.0.0.1:8000/predict/?model=${model}`,
        {
            method: "POST",
            body: formData,
        }
    );

    const data = await response.json();
    const prediction = data.label;

    predictionText.innerText = `Predicted Label: ${prediction}`;
}

function updateImagePreview() {
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        const imagePlaceholder = document.getElementById("image-placeholder");
        imagePlaceholder.src = e.target.result;
    };

    reader.readAsDataURL(file);
}

function onClickUpload() {
    document.getElementById("file-input").click();
}
