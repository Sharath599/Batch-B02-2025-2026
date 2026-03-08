# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")
# @app.route("/signup")
# def signup():
#     return render_template("signup.html")
# @app.route("/signin")
# def signin():
#     return render_template("signin.html")
# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for
# import sqlite3

# app = Flask(__name__)

# # ---------- DATABASE CONNECTION ----------
# def get_db_connection():
#     conn = sqlite3.connect("signup.db")
#     conn.row_factory = sqlite3.Row
#     return conn

# # ---------- CREATE TABLE ----------
# def create_table():
#     conn = get_db_connection()
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE,
#             fullname TEXT,
#             email TEXT UNIQUE,
#             mobile TEXT,
#             password TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# create_table()

# # ---------- ROUTES ----------
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         username = request.form["username"]
#         fullname = request.form["fullname"]
#         email = request.form["email"]
#         mobile = request.form["mobile"]
#         password = request.form["password"]

#         conn = get_db_connection()
#         try:
#             conn.execute("""
#                 INSERT INTO users (username, fullname, email, mobile, password)
#                 VALUES (?, ?, ?, ?, ?)
#             """, (username, fullname, email, mobile, password))
#             conn.commit()
#         except sqlite3.IntegrityError:
#             return "Username or Email already exists"
#         finally:
#             conn.close()

#         return redirect(url_for("signin"))

#     return render_template("signup.html")

# @app.route("/signin")
# def signin():
#     return render_template("signin.html")

# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session
# import sqlite3

# app = Flask(__name__)
# app.secret_key = "ecovision_secret_key"

# # ---------- DB ----------
# def get_db_connection():from flask import Flask, render_template, request, redirect, url_for, session

#     conn = sqlite3.connect("signup.db")
#     conn.row_factory = sqlite3.Row
#     return conn

# # ---------- ROUTES ----------
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         data = (
#             request.form["username"],
#             request.form["fullname"],
#             request.form["email"],
#             request.form["mobile"],
#             request.form["password"]
#         )

#         conn = get_db_connection()
#         try:
#             conn.execute(
#                 "INSERT INTO users (username, fullname, email, mobile, password) VALUES (?, ?, ?, ?, ?)",
#                 data
#             )
#             conn.commit()
#         except:
#             return "User already exists"
#         finally:
#             conn.close()

#         return redirect("/signin")

#     return render_template("signup.html")

# @app.route("/signin", methods=["GET", "POST"])
# def signin():
#     if request.method == "POST":
#         username = request.form["username"]
#         password = request.form["password"]

#         conn = get_db_connection()
#         user = conn.execute(
#             "SELECT * FROM users WHERE username=? AND password=?",
#             (username, password)
#         ).fetchone()
#         conn.close()

#         if user:
#             session["user"] = user["username"]
#             return redirect("/dashboard")
#         else:
#             return "Invalid username or password"

#     return render_template("signin.html")

# @app.route("/dashboard")
# def dashboard():
#     if "user" not in session:
#         return redirect("/signin")
#     return render_template("dashboard.html", user=session["user"])

# @app.route("/logout")
# def logout():
#     session.clear()
#     return redirect("/")
# @app.route("/explore")
# def explore():
#     # protect page (only logged-in users)
#     # if "username" not in session:
#     #     return redirect(url_for("signin"))

#     return render_template("explore.html")
# @app.route("/prediction")
# def prediction():
#     # if "username" not in session:
#     #     return redirect(url_for("signin"))
#     return render_template("prediction.html")


# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "ecovision_secret_key"

device = torch.device("cpu")

# ==========================
# DATABASE
# ==========================
def get_db_connection():
    conn = sqlite3.connect("signup.db")
    conn.row_factory = sqlite3.Row
    return conn


# ==========================
# LOAD MODEL
# ==========================
class EcoVision(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.img_fc = nn.Linear(32*32*32,128)

        self.env_fc = nn.Sequential(
            nn.Linear(3,32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(160,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )

    def forward(self,img,env):
        img_feat = self.cnn(img)
        img_feat = self.img_fc(img_feat)
        env_feat = self.env_fc(env)
        combined = torch.cat((img_feat,env_feat),dim=1)
        return self.classifier(combined)


checkpoint = torch.load("ecovision_model.pth", map_location=device, weights_only=False)
label_encoder = checkpoint["label_encoder"]
scaler = checkpoint["scaler"]

model = EcoVision(len(label_encoder.classes_))
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# ==========================
# ROUTES
# ==========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = (
            request.form["username"],
            request.form["fullname"],
            request.form["email"],
            request.form["mobile"],
            request.form["password"]
        )

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (username, fullname, email, mobile, password) VALUES (?, ?, ?, ?, ?)",
                data
            )
            conn.commit()
        except:
            return "User already exists"
        finally:
            conn.close()

        return redirect("/signin")

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session["user"] = user["username"]
            return redirect("/dashboard")
        else:
            return "Invalid username or password"

    return render_template("signin.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/signin")
    return render_template("dashboard.html", user=session["user"])


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user" not in session:
        return redirect("/signin")

    result = None

    if request.method == "POST":
        file = request.files["image"]

        humidity = request.form.get("humidity")
        wind = request.form.get("wind_speed")
        wetness = request.form.get("wetness")

        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0)

        # Optional environmental parameters
        if humidity and wind and wetness:
            env = np.array([[float(humidity), float(wind), float(wetness)]])
            env = scaler.transform(env)
        else:
            env = np.zeros((1,3))

        env = torch.tensor(env, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(image, env)
            _, predicted = torch.max(outputs, 1)
            result = label_encoder.inverse_transform([predicted.item()])[0]

    return render_template("prediction.html", prediction=result)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
