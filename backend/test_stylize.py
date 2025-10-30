import requests

url = "http://127.0.0.1:5000/stylize"

files = {
    "content": open(r"C:\Users\iyedm\OneDrive\Desktop\StealStyle\backend\uploads\louvre.jpg", "rb"),
    "style":   open(r"C:\Users\iyedm\OneDrive\Desktop\StealStyle\backend\uploads\starry_night.jpg", "rb"),
}

response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
