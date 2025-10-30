# 🎨 Neural Style Transfer (NST) Web App

![NST Banner](https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/content.jpg)

Bring your photos to life with AI! This project allows you to **transform your content images** into stunning artwork in the style of famous paintings using **Neural Style Transfer**.  

---

## **Demo: See It in Action**

<div style="display: flex; justify-content: center; gap: 30px; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <h4>Content Image</h4>
    <img src="https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/content.jpg" width="200px" alt="Content" />
    <p>Original photo</p>
  </div>

  <div style="text-align: center;">
    <h4>Style Image</h4>
    <img src="https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/style.jpg" width="200px" alt="Style" />
    <p>Starry Night style</p>
  </div>

  <div style="text-align: center;">
    <h4>Generated Image</h4>
    <img src="https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/generated.png" width="200px" alt="Generated" />
    <p>Stylized output</p>
  </div>

</div>

<br/>

---

## **🚀 Features**

- Upload your **content** and **style** images
- Generates multiple intermediate stylized images
- Built with **Python (TensorFlow / Keras)** for NST backend
- **Flask API** serves images to a **React frontend**
- Interactive, user-friendly UI
- Easily extendable for custom styles

---

## **💻 Project Structure**

```text
nst-app/                 # root folder
├── backend/             # Flask backend
│   ├── app.py           # main Flask server
│   ├── requirements.txt # Python dependencies
│   ├── uploads/         # uploaded and generated images
│   └── nst/             # NST code
│       ├── __init__.py
│       ├── model.py
│       ├── preprocess.py
│       ├── gram.py
│       ├── losses.py
│       └── transfer.py
├── frontend/            # React frontend
│   ├── package.json
│   ├── public/
│   └── src/
└── README.md
