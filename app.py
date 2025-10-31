from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import sqlite3, qrcode, io, json, pandas as pd, os, base64, pickle, cv2
from datetime import datetime
import numpy as np
from deepface import DeepFace

app = Flask(__name__)
app.secret_key = "secret123"
DB = "attendance.db"
EXCEL_FILE = "attendance.xlsx"

# ===================================================================
# ✅ FIXED: Use relative model path (works locally + Render)
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl")

# similarity threshold
SIMILARITY_THRESHOLD = 0.60

# ===================================================================
# LOAD FACE MODEL
# ===================================================================
def load_face_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    embeddings, names = [], []

    if isinstance(raw, dict):
        if "embeddings" in raw and "names" in raw:
            embeddings = [np.array(e, dtype="float32") for e in raw["embeddings"]]
            names = list(raw["names"])
        elif "encodings" in raw and "names" in raw:
            embeddings = [np.array(e, dtype="float32") for e in raw["encodings"]]
            names = list(raw["names"])
        else:
            for k, v in raw.items():
                names.append(str(k))
                embeddings.append(np.array(v, dtype="float32"))
    elif isinstance(raw, list):
        for item in raw:
            identity = item.get("identity") or item.get("id") or item.get("image") or ""
            try:
                name = os.path.basename(os.path.dirname(identity)) or os.path.basename(identity).split("_")[0]
            except Exception:
                name = str(identity)
            emb = item.get("embedding") or item.get("emb") or item.get("vector")
            if emb is not None:
                embeddings.append(np.array(emb, dtype="float32"))
                names.append(name)
    else:
        raise ValueError("Unsupported model structure.")

    if not embeddings:
        raise ValueError("No embeddings found in model file.")

    embeddings = np.vstack([e / (np.linalg.norm(e) + 1e-10) for e in embeddings])
    return {"embeddings": embeddings, "names": names}


try:
    FACE_MODEL = load_face_model(MODEL_PATH)
    print(f"[INFO] Loaded face model with {len(FACE_MODEL['names'])} identities.")
except Exception as e:
    FACE_MODEL = None
    print(f"[ERROR] Could not load model: {e}")

# ===================================================================
# UTILITIES
# ===================================================================
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

def get_embedding_from_image_bgr(bgr_image):
    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rep = DeepFace.represent(rgb, model_name="Facenet", enforce_detection=True)
        if isinstance(rep, list) and len(rep) > 0:
            emb = np.array(rep[0]["embedding"], dtype="float32")
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            return emb
        return None
    except Exception as e:
        print("[WARN] DeepFace failed to get embedding:", e)
        return None

def predict_from_b64(image_b64):
    if FACE_MODEL is None:
        print("[ERROR] Face model not loaded; prediction skipped.")
        return None, 0.0
    try:
        header, b64data = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        img_bytes = base64.b64decode(b64data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, 0.0

        emb = get_embedding_from_image_bgr(img)
        if emb is None:
            return None, 0.0

        model_embs = FACE_MODEL["embeddings"]
        sims = model_embs.dot(emb)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_name = FACE_MODEL["names"][best_idx]
        return best_name, best_score
    except Exception as e:
        print("[ERROR] predict_from_b64 failed:", e)
        return None, 0.0

# ===================================================================
# DATABASE
# ===================================================================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  roll_no TEXT UNIQUE,
                  name TEXT,
                  password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  roll_no TEXT,
                  name TEXT,
                  date TEXT,
                  time TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS admins
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')
    predefined_students = [
        ("101", "Ayush Singh"),
        ("102", "Rohan Kumar"),
        ("103", "Priya Sharma"),
        ("104", "Ankit Verma"),
        ("105", "Neha Gupta"),
        ("106", "Vikas Yadav"),
        ("107", "Simran Kaur"),
        ("108", "Rahul Sharma"),
        ("109", "Sneha Patel"),
        ("110", "Arjun Mehta")
    ]
    for roll, name in predefined_students:
        password = name[:4].upper() + "123"
        c.execute("SELECT * FROM students WHERE roll_no=?", (roll,))
        if not c.fetchone():
            c.execute("INSERT INTO students (roll_no, name, password) VALUES (?, ?, ?)", (roll, name, password))
    predefined_admins = [("admin1", "admin123"), ("admin2", "admin456")]
    for username, password in predefined_admins:
        c.execute("SELECT * FROM admins WHERE username=?", (username,))
        if not c.fetchone():
            c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def export_to_excel():
    try:
        conn = sqlite3.connect(DB)
        df = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()
        if not df.empty:
            df.to_excel(EXCEL_FILE, index=False)
        else:
            pd.DataFrame(columns=["id","roll_no","name","date","time"]).to_excel(EXCEL_FILE, index=False)
    except Exception as e:
        print("⚠ Excel export failed:", e)

init_db()

# ===================================================================
# ROUTES
# ===================================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        name = request.form["name"].strip()
        password = request.form["password"].strip()
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM students WHERE name=? AND password=?", (name, password))
        student = c.fetchone()
        conn.close()
        if student:
            session["student"] = {"id": student[0], "roll_no": student[1], "name": student[2]}
            return redirect(url_for("student_dashboard"))
        return render_template("login.html", error="Invalid student credentials!")
    return render_template("login.html")

@app.route("/admin_login", methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM admins WHERE username=? AND password=?", (username, password))
        admin = c.fetchone()
        conn.close()
        if admin:
            session["admin"] = {"id": admin[0], "username": admin[1]}
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Invalid admin credentials!")
    return render_template("admin_login.html")

@app.route("/student")
def student_dashboard():
    if "student" not in session:
        return redirect(url_for("login"))
    student = session["student"]
    qr_folder = os.path.join(app.root_path, 'static', 'qr')
    os.makedirs(qr_folder, exist_ok=True)
    qr_filename = f"{student['roll_no']}.png"
    qr_path = os.path.join(qr_folder, qr_filename)
    img = qrcode.make(json.dumps({"roll_no": student["roll_no"], "name": student["name"]}))
    img.save(qr_path)
    return render_template("student.html", student=student, qr_file=url_for('static', filename='qr/' + qr_filename))

@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY id DESC")
    records = c.fetchall()
    conn.close()
    return render_template("admin.html", records=records, admin=session["admin"])

@app.route("/verify_face_and_qr", methods=["POST"])
def verify_face_and_qr():
    if "admin" not in session:
        return jsonify({"status":"error","message":"not authorized"}), 401
    data = request.get_json()
    if not data:
        return jsonify({"status":"error","message":"No JSON received"}), 400

    qr_json_str = data.get("qr_data")
    image_b64 = data.get("image")
    if not qr_json_str or not image_b64:
        return jsonify({"status":"error","message":"qr_data and image required"}), 400

    qr = json.loads(qr_json_str)
    qr_name = qr.get("name","").strip()
    qr_roll = qr.get("roll_no","").strip()
    predicted_name, score = predict_from_b64(image_b64)

    if predicted_name and qr_name and predicted_name.lower() == qr_name.lower() and score >= SIMILARITY_THRESHOLD:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%H:%M:%S")
        c.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (qr_roll, date))
        if c.fetchone():
            conn.close()
            return jsonify({"status":"error","message":"Already marked today"})
        c.execute("INSERT INTO attendance (roll_no, name, date, time) VALUES (?, ?, ?, ?)", (qr_roll, qr_name, date, time))
        conn.commit()
        conn.close()
        export_to_excel()
        return jsonify({"status":"success","message":f"✅ Attendance marked for {qr_name}", "score": score})
    else:
        return jsonify({"status":"error","message":f"Face not matched (predicted={predicted_name}, score={score:.3f})"})

@app.route("/export")
def export_excel():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    export_to_excel()
    if os.path.exists(EXCEL_FILE):
        return send_file(EXCEL_FILE, as_attachment=True)
    return "No attendance data"

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    if FACE_MODEL is None:
        raise RuntimeError(f"❌ Face model not loaded! Upload it to {MODEL_PATH} before running.")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
