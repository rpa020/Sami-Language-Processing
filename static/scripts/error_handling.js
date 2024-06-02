function validate_input(event) {

    var text = document.getElementById("text_input").value;
    var error = document.getElementById("error_message");

    if (!text) {
        error.innerText = "* Please provide text or upload a file";
        error.style.display = "block";
        event.preventDefault();
    }else
        error.style.display = "none";
}

function handleFileUpload(event) {
    
    var file = event.target.files[0];
    var error = document.getElementById("error_message");

    if (file) {
        if(file.size > 2 * 1024 * 1024){
            error.innerText = "* File size exceeds 2 MB";
            error.style.display = "block";
            event.preventDefault();
        }else{
            error.style.display = "none";
            var reader = new FileReader();
            reader.onload = function(e) {
                var content = e.target.result;
                var textarea = document.getElementById("text_input");
                textarea.value = content;
            };
            reader.readAsText(file);
        }
    }
}

document.addEventListener("DOMContentLoaded", function() {
    var fileInput = document.getElementById("uploadedFile");
    fileInput.addEventListener("change", handleFileUpload);
});
