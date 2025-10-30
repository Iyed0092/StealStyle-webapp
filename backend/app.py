import os
from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from nst.transfer import stylize_from_paths
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/stylize", methods=["POST"])
def stylize_image():
    try:
        content_file = request.files.get("content")
        style_file = request.files.get("style")
        if not content_file or not style_file:
            return jsonify({"error": "Missing content or style file"}), 400

        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)
        content_path = os.path.join(app.config["UPLOAD_FOLDER"], content_filename)
        style_path = os.path.join(app.config["UPLOAD_FOLDER"], style_filename)
        content_file.save(content_path)
        style_file.save(style_path)

        output_dir = os.path.join(app.config["UPLOAD_FOLDER"], "generated_images")
        loss_dir = os.path.join(app.config["UPLOAD_FOLDER"], "loss_data")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)


        saved_images = stylize_from_paths(
            content_path,
            style_path
        )

        if not saved_images:
            return jsonify({"error": "No images generated"}), 500

        image_urls = [
            os.path.relpath(p, app.root_path).replace(os.path.sep, "/")
            for p in saved_images
        ]

        return Response(
            response=json.dumps({"images": image_urls}),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        import traceback
        return Response(
            response=json.dumps({
                "error": str(e),
                "trace": traceback.format_exc()
            }),
            status=500,
            mimetype="application/json"
        )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
