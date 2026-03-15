from flask import Flask, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- DATABASE ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- UPLOAD FOLDER ----------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- USER MODEL ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# ---------------- PREDICTION MODEL ----------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    disease = db.Column(db.String(100))
    percentage = db.Column(db.String(20))
    image_path = db.Column(db.String(200))  # store uploaded image path
    date = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("plant_model.h5")
classes = ['Leaf Blight', 'Mosaic Virus', 'Powdery Mildew',
           'healthy', 'leaf curls', 'leaf spot', 'shot hole disease']

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session["username"] = username
            return redirect("/dashboard")
        else:
            flash("Invalid Username or Password")
    return render_template("login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match")
            return redirect("/register")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists")
            return redirect("/register")

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.")
        return redirect("/")
    return render_template("register.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect("/")

    predictions = Prediction.query.filter_by(username=session["username"]) \
                    .order_by(Prediction.date.desc()).all()

    return render_template("dashboard.html",
                           username=session["username"],
                           predictions=predictions)

# ---------------- IMAGE UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    if "username" not in session:
        return redirect("/")

    if "image" not in request.files:
        flash("No file selected")
        return redirect("/dashboard")

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected")
        return redirect("/dashboard")

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ---------------- PREDICTION ----------------
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))  # Replace with your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction)) * 100
    disease = classes[np.argmax(prediction)]

    # Save prediction
    new_prediction = Prediction(
        username=session["username"],
        disease=disease,
        percentage=f"{confidence:.2f}%",
        image_path=filepath
    )
    db.session.add(new_prediction)
    db.session.commit()

    flash(f"Image uploaded! Prediction: {disease} ({confidence:.2f}%)")
    return redirect("/dashboard")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect("/")

# ---------------- RUN ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)