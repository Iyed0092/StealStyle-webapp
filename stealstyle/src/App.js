import React, { useState } from "react";
import "./App.css";

function App() {
  const [contentFile, setContentFile] = useState(null);
  const [styleFile, setStyleFile] = useState(null);
  const [resultImages, setResultImages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!contentFile || !styleFile) {
      alert("Please select both content and style images!");
      return;
    }

    setLoading(true);
    setResultImages([]);

    const formData = new FormData();
    formData.append("content", contentFile);
    formData.append("style", styleFile);

    try {
      const res = await fetch("http://localhost:5000/stylize", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to generate image");

      const data = await res.json();

      if (data.images && data.images.length > 0) {
        const backendUrl = "http://localhost:5000/";
        const imagesWithFullUrl = data.images.map((img) =>
          img.startsWith("http") ? img : `${backendUrl}${img}`
        );
        setResultImages(imagesWithFullUrl);
      } else {
        throw new Error("No images returned from server");
      }
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image")) {
      if (type === "content") setContentFile(file);
      else setStyleFile(file);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const renderPreview = (file, title) =>
    file && (
      <div className="preview-box">
        <h3>{title}</h3>
        <img src={URL.createObjectURL(file)} alt={title} />
      </div>
    );

  return (
    <div className="app-container">
      <h1>🎨 Neural Style Transfer</h1>
      <p className="subtitle">
        Upload or drag & drop your images to create stunning art!
      </p>

      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="file-inputs">
          {["content", "style"].map((type) => (
            <div
              key={type}
              className="file-wrapper drop-zone"
              onDrop={(e) => handleDrop(e, type)}
              onDragOver={handleDragOver}
            >
              <label>{type.charAt(0).toUpperCase() + type.slice(1)} Image</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) =>
                  type === "content"
                    ? setContentFile(e.target.files[0])
                    : setStyleFile(e.target.files[0])
                }
              />
              {type === "content" && contentFile && (
                <p className="filename">{contentFile.name}</p>
              )}
              {type === "style" && styleFile && (
                <p className="filename">{styleFile.name}</p>
              )}
            </div>
          ))}
        </div>
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? "Generating..." : "Generate Image"}
        </button>
      </form>

      {loading && (
        <div className="loader">
          <div className="spinner"></div>
          <p>Processing your image...</p>
        </div>
      )}

      <div className="preview-container">
        {renderPreview(contentFile, "Content")}
        {renderPreview(styleFile, "Style")}

 {resultImages.length > 0 && (
  <div className="generated-gallery">
    <h3>Generated Images</h3>
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "10px",
      }}
    >
      {resultImages.map((img, idx) => {
        let epochNumber = img.match(/generated_(\d+)\./);
        epochNumber = epochNumber ? parseInt(epochNumber[1]) : idx;

        return (
          <div
            key={idx}
            style={{ textAlign: "center", border: "1px solid #ccc", padding: "5px" }}
          >
            <img
              src={img}
              alt={`Generated ${epochNumber}`}
              style={{ width: "200px", height: "auto" }}
            />
            <div>Epoch {epochNumber}</div>
          </div>
        );
      })}
    </div>
  </div>
)}

      </div>
    </div>
  );
}

export default App;
