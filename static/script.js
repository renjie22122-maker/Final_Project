let images = [];
let currentIndex = 0;
let historyList = [];

function uploadImages() {
    let formData = new FormData();
    let files = document.getElementById("uploadInput").files;

    for (let file of files) {
        formData.append("file", file);
    }

    $.ajax({
        url: "/upload",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            images = response.filenames;
            currentIndex = 0;
            updatePreview();
        },
        error: function () {
            alert("上传失败！");
        }
    });
}

function updatePreview() {
    if (images.length === 0) {
        $("#previewOriginal").attr("src", "").hide();
        $("#previewStyled").attr("src", "").hide();
    } else {
        const originalPath = "/uploads/" + images[currentIndex];
        $("#previewOriginal").attr("src", originalPath).show();
        $("#previewStyled").attr("src", "").hide();
        $("#downloadBtn").hide();
    }
}

function prevImage() {
    if (currentIndex > 0) {
        currentIndex--;
        updatePreview();
    }
}

function nextImage() {
    if (currentIndex < images.length - 1) {
        currentIndex++;
        updatePreview();
    }
}

function applyStyle() {
    if (images.length === 0) {
        alert("请先上传图片！");
        return;
    }

    const direction = $("input[name='style_mode']:checked").val();
    const modelName = $("#modelSelect").val();

    $.ajax({
        url: "/apply_style",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            filename: images[currentIndex],
            direction: direction,
            model: modelName
        }),
        success: function (response) {
            const styledImage = "/outputs/" + response.output_filename;
            $("#previewStyled").attr("src", styledImage).show();
            $("#downloadBtn").attr("href", styledImage).show().text("📥 下载");
            updateHistory(images[currentIndex], response.output_filename, direction, modelName);
        },
        error: function () {
            alert("转换失败！");
        }
    });
}

function updateHistory(inputImage, outputImage, direction, model) {
    const dirLabel = (direction === "m2r") ? "Monet → Real" : "Real → Monet";
    const entry = {
        input: inputImage,
        output: outputImage,
        direction: dirLabel,
        model: model
    };
    historyList.unshift(entry);
    renderHistory();
}

function renderHistory() {
    let html = "";
    historyList.forEach(entry => {
        html += `
            <li>
                <strong>${entry.model}</strong> [${entry.direction}]: 
                <img src="/uploads/${entry.input}" width="80"> ➝ 
                <img src="/outputs/${entry.output}" width="80">
            </li>
        `;
    });
    $("#historyList").html(html);
}
