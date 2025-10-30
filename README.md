# 🎨 Neural Style Transfer (NST) Web App

![NST Banner](https://user-images.githubusercontent.com/your-image-placeholder/nst-banner.png)

Bring your photos to life with AI! This project allows you to **transform your content images** into stunning artwork in the style of famous paintings using **Neural Style Transfer**.  

---

## **Demo: See It in Action**

<div align="center">
  <h4>Content Image</h4>
  <img src="https://github.com/Iyed0092/StealStyle-webapp/raw/main/backend/uploads/louvre.jpg" width="30%" alt="Content Image" />

  <h4>Style Image</h4>
  <img src="https://github.com/Iyed0092/StealStyle-webapp/raw/main/backend/uploads/starry_night.jpg" width="30%" alt="Style Image" />

  <h4>Generated Image</h4>
  <img src="https://github.com/Iyed0092/StealStyle-webapp/raw/main/backend/uploads/generated_images/generated_1000.png" width="30%" alt="NST Example" />
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

```

🎯 Usage

- Open the web app in your browser: http://localhost:3000

- Upload a content image and a style image

- Click Generate Image

- See the gallery of generated images below your uploads

💡 Notes

- Intermediate images are saved in backend/uploads/generated_images

- You can adjust NST parameters (epochs, alpha, beta) in transfer.py

- The app is fully extensible for multiple styles, larger images, or live previews

🛠 Tech Stack

Backend: Python, Flask, TensorFlow, Keras

Frontend: React, JavaScript, HTML, CSS

Style Transfer: Neural Style Transfer (Gatys et al., 2015)