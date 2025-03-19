function previewImage() {
    let fileInput = document.getElementById("upload");
    let file = fileInput.files[0];
    if (file) {
        let preview = document.getElementById("preview");
        let objectUrl = URL.createObjectURL(file);
        preview.src = objectUrl;
        preview.style.display = "block";
        preview.onload = function() {
            URL.revokeObjectURL(objectUrl);
        };
    }
}

function predictSeverity() {
    let fileInput = document.getElementById("upload");
    let file = fileInput.files[0];
    if (!file) {
        alert("Please select an X-ray image first.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    document.getElementById("result").innerText = "Predicting...";
    document.getElementById("solution").innerText = "";
    document.getElementById("severityImage").style.display = "none";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
            return;
        }

        document.getElementById("result").innerText = "Prediction: " + data.prediction;
        document.getElementById("solution").innerText = "Solution: " + data.solution;
        
        let severityImage = document.getElementById("severityImage");
        severityImage.src = data.severity_image + "?" + new Date().getTime();
        severityImage.style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error in prediction.";
    });
}
