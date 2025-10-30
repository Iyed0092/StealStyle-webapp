# 🎨 Neural Style Transfer (NST) Web App

![NST Banner](https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/content.jpg)

Bring your photos to life with AI! This project allows you to **transform your content images** into stunning artwork in the style of famous paintings using **Neural Style Transfer**.  

---

### Demo: See It in Action

**Content Image**  
![Content](https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/louvre.jpg)

**Style Image**  
![Style](https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/starry_night.jpg)

**Generated Image**  
![Generated](https://github.com/Iyed0092/StealStyle-webapp/raw/main/assets/demo-images/generated_1000.png)


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
