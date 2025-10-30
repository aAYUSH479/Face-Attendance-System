# app.py
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import sqlite3, qrcode, io, json, pandas as pd, os, base64, pickle, cv2
from datetime import datetime
import numpy as np

# DeepFace used to extract FaceNet embeddings from incoming images
from deepface import DeepFace

app = Flask(__name__)
app.secret_key = "secret123"
DB = "attendance.db"
EXCEL_FILE = "attendance.xlsx"

# <-- POINT THIS to your existing pkl -->
MODEL_PATH = r"C:\Users\ayush\OneDrive\Desktop\New folder\models\ds_model_facenet_detector_opencv_aligned_normalization_base_expand_0.pkl"

# similarity threshold for FaceNet cosine similarity (tune if needed)
SIMILARITY_THRESHOLD = 0.60

# ---------- load model (supports the 'list of dicts' format you showed) ----------
def load_face_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)

    # handle a few possible formats
    embeddings = []
    names = []

    # If it's a dict with keys like 'embeddings' or 'encodings'
    if isinstance(raw, dict):
        # try common keys
        if "embeddings" in raw and "names" in raw:
            embeddings = [np.array(e, dtype="float32") for e in raw["embeddings"]]
            names = list(raw["names"])
        elif "encodings" in raw and "names" in raw:
            embeddings = [np.array(e, dtype="float32") for e in raw["encodings"]]
            names = list(raw["names"])
        else:
            # maybe the dict maps identity -> embedding
            for k, v in raw.items():
                names.append(str(k))
                embeddings.append(np.array(v, dtype="float32"))
    # If file is a list of records (your sample showed that)
    elif isinstance(raw, list):
        for item in raw:
            # each item looks like {'identity': 'C:\\..\\Rohan Kumar\\file.jpg', 'embedding': [...], ...}
            identity = item.get("identity") or item.get("id") or item.get("image") or ""
            # derive name as parent folder name first, fallback to filename part
            name = ""
            try:
                name = os.path.basename(os.path.dirname(identity)) or os.path.basename(identity).split("_")[0]
            except Exception:
                name = str(identity)
            emb = item.get("embedding") or item.get("emb") or item.get("vector") or None
            if emb is None:
                # maybe nested differently
                for k in ("embedding", "emb", "vector"):
                    if k in item:
                        emb = item[k]
                        break
            if emb is None:
                continue
            embeddings.append(np.array(emb, dtype="float32"))
            names.append(name)
    else:
        raise ValueError("Unsupported model structure: must be dict or list")

    if len(embeddings) == 0:
        raise ValueError("No embeddings found in model file.")

    # normalize embeddings (important for cosine)
    embeddings = np.vstack([e / (np.linalg.norm(e) + 1e-10) for e in embeddings])
    return {"embeddings": embeddings, "names": names}

# load at app start (will crash with explicit error if file missing)
try:
    FACE_MODEL = load_face_model(MODEL_PATH)
    print(f"[INFO] Loaded face model: {len(FACE_MODEL['names'])} identities.")
except Exception as e:
    FACE_MODEL = None
    print(f"[ERROR] Could not load model: {e}")

# ---------- helper functions ----------
def cosine_similarity(a, b):
    # a, b are 1D numpy arrays, both normalized
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

def get_embedding_from_image_bgr(bgr_image):
    """
    Accepts an OpenCV BGR image (numpy array)
    Returns embedding using DeepFace (Facenet).
    """
    try:
        # DeepFace expects either a file path or numpy array (RGB) — pass RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rep = DeepFace.represent(rgb, model_name="Facenet", enforce_detection=True)
        # DeepFace.represent returns list of dicts (one per face); we'll take the first
        if isinstance(rep, list) and len(rep) > 0:
            emb = np.array(rep[0]["embedding"], dtype="float32")
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            return emb
        # fallback
        return None
    except Exception as e:
        # if DeepFace fails (detection issues), bubble error for debugging
        print("[WARN] DeepFace failed to get embedding:", e)
        return None

def predict_from_b64(image_b64):
    """
    Decode base64 DataURL -> OpenCV image -> compute embedding -> compare to model -> best match and score
    Returns (best_name_or_None, best_score_float)
    """
    if FACE_MODEL is None:
        return None, 0.0

    try:
        header, b64data = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        img_bytes = base64.b64decode(b64data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, 0.0

        # get embedding
        emb = get_embedding_from_image_bgr(img)
        if emb is None:
            # fallback: try face_recognition 128D encoding and compare by L2 with model - but likely incompatible — return no match
            return None, 0.0

        # compare against model embeddings (which are normalized)
        model_embs = FACE_MODEL["embeddings"]  # shape (N, D)
        sims = model_embs.dot(emb)  # cos similarity because both sides normalized
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_name = FACE_MODEL["names"][best_idx]
        return best_name, best_score
    except Exception as e:
        print("[ERROR] predict_from_b64 failed:", e)
        return None, 0.0

# ---------- DB / existing routes (unchanged) ----------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # Student table
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  roll_no TEXT UNIQUE,
                  name TEXT,
                  password TEXT)''')
    # Attendance table
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  roll_no TEXT,
                  name TEXT,
                  date TEXT,
                  time TEXT)''')
    # Admins table
    c.execute('''CREATE TABLE IF NOT EXISTS admins
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')
    # Predefined students (keeps your existing entries)
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
    # Predefined admins
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
            df = pd.DataFrame(columns=["id","roll_no","name","date","time"])
            df.to_excel(EXCEL_FILE, index=False)
    except Exception as e:
        print("⚠ Excel export failed:", e)

# init DB
init_db()

# ---------- WEB ROUTES ----------
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
        else:
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
        else:
            return render_template("admin_login.html", error="Invalid admin credentials!")
    return render_template("admin_login.html")

@app.route("/student")
def student_dashboard():
    if "student" not in session:
        return redirect(url_for("login"))
    try:
        student = session["student"]
        data = {"roll_no": student["roll_no"], "name": student["name"]}
        qr_folder = os.path.join(app.root_path, 'static', 'qr')
        os.makedirs(qr_folder, exist_ok=True)
        qr_filename = f"{student['roll_no']}.png"
        qr_path = os.path.join(qr_folder, qr_filename)
        img = qrcode.make(json.dumps(data))
        img.save(qr_path)
        return render_template("student.html", student=student, qr_file=url_for('static', filename='qr/' + qr_filename))
    except Exception as e:
        return f"QR Generation Error: {str(e)}"

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

# Keep your original mark_attendance route intact
@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    if "admin" not in session:
        return jsonify({"status":"error","message":"not authorized"}), 401
    try:
        if request.is_json:
            payload = request.get_json()
            qr_data = payload.get("qr_data")
        else:
            qr_data = request.form.get("qr_data") or request.values.get("qr_data")
        student = json.loads(qr_data)
        roll_no = student.get("roll_no")
        name = student.get("name")
    except Exception as e:
        return jsonify({"status":"error","message":f"Invalid QR data: {str(e)}"}), 400

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    c.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (roll_no, date))
    existing = c.fetchone()
    if existing:
        conn.close()
        return jsonify({"status":"error", "message":"Attendance already marked today!"})

    c.execute("INSERT INTO attendance (roll_no, name, date, time) VALUES (?,?,?,?)",
              (roll_no, name, date, time))
    conn.commit()
    conn.close()

    try:
        export_to_excel()
    except Exception as e:
        print("⚠ Attendance saved but Excel export failed:", e)

    return jsonify({"status":"ok","message":"Attendance marked successfully", "roll_no": roll_no, "name": name})

# ---------- NEW: verify face + qr endpoint (JSON from frontend) ----------
@app.route("/verify_face_and_qr", methods=["POST"])
def verify_face_and_qr():
    # require admin session like your other code
    if "admin" not in session:
        return jsonify({"status":"error","message":"not authorized"}), 401

    try:
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
            # mark attendance
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")
            c.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (qr_roll, date))
            if c.fetchone():
                conn.close()
                return jsonify({"status":"error","message":"Already marked today"})
            c.execute("INSERT INTO attendance (roll_no, name, date, time) VALUES (?, ?, ?, ?)",
                      (qr_roll, qr_name, date, time))
            conn.commit()
            conn.close()
            export_to_excel()
            return jsonify({"status":"success","message":f"✅ Attendance marked for {qr_name}", "score": score})
        else:
            return jsonify({"status":"error","message":f"Face not matched (predicted={predicted_name}, score={score:.3f})"})
    except Exception as e:
        return jsonify({"status":"error","message":f"Error: {str(e)}"}), 500

# ---------- NEW: detect_face endpoint (multipart/form-data or JSON) ----------
@app.route("/detect_face", methods=["POST"])
def detect_face():
    """
    Accepts:
      - multipart form with 'image' file (blob) OR
      - JSON with 'image' base64 (dataURL).
    Optional:
      - 'qr_data' (stringified JSON) in form or JSON, if provided we will try to mark attendance when matched.
    """
    if "admin" not in session:
        return jsonify({"status":"error","message":"not authorized"}), 401

    try:
        # image input
        if request.content_type and "application/json" in request.content_type:
            payload = request.get_json()
            image_b64 = payload.get("image")
            qr_json = payload.get("qr_data")
        else:
            # form-data
            image_file = request.files.get("image")
            qr_json = request.form.get("qr_data")
            if image_file:
                # convert to base64 dataURL
                bytes_data = image_file.read()
                image_b64 = "data:image/jpeg;base64," + base64.b64encode(bytes_data).decode("utf-8")
            else:
                return jsonify({"status":"error","message":"No image provided"}), 400

        predicted_name, score = predict_from_b64(image_b64)

        # if qr provided, try to match and mark attendance
        if qr_json:
            try:
                qr = json.loads(qr_json)
                qr_name = qr.get("name","").strip()
                qr_roll = qr.get("roll_no","").strip()
            except Exception:
                qr = None
                qr_name = qr_roll = None

            if predicted_name and qr_name and predicted_name.lower() == qr_name.lower() and score >= SIMILARITY_THRESHOLD:
                # mark attendance
                conn = sqlite3.connect(DB)
                c = conn.cursor()
                date = datetime.now().strftime("%Y-%m-%d")
                time = datetime.now().strftime("%H:%M:%S")
                c.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (qr_roll, date))
                if c.fetchone():
                    conn.close()
                    return jsonify({"status":"error","message":"Already marked today", "name": predicted_name, "score": score})
                c.execute("INSERT INTO attendance (roll_no, name, date, time) VALUES (?, ?, ?, ?)",
                          (qr_roll, qr_name, date, time))
                conn.commit()
                conn.close()
                export_to_excel()
                return jsonify({"status":"success","message":f"✅ Attendance marked for {qr_name}", "name": predicted_name, "score": score})
            else:
                return jsonify({"status":"error","message":f"Face mismatch (predicted={predicted_name}, score={score:.3f})", "name": predicted_name, "score": score})
        else:
            # return predicted name/score for client-side decision
            if predicted_name:
                return jsonify({"status":"ok","name": predicted_name, "score": score})
            return jsonify({"status":"error","message":"No face found or embedding failed"})
    except Exception as e:
        return jsonify({"status":"error","message":f"Error: {str(e)}"}), 500

# ---------- rest of your unchanged routes ----------
@app.route("/export")
def export_excel():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    export_to_excel()
    if os.path.exists(EXCEL_FILE):
        return send_file(EXCEL_FILE, as_attachment=True)
    return "No attendance data"

@app.route("/clear_attendance")
def clear_attendance():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    if os.path.exists(EXCEL_FILE):
        os.remove(EXCEL_FILE)
    export_to_excel()
    return redirect(url_for("admin_dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    # sanity: ensure model is loaded (log message already printed earlier)
    if FACE_MODEL is None:
        print("[WARN] Face model not loaded. /verify_face_and_qr and /detect_face will not work until MODEL_PATH is valid.")
    app.run(host='0.0.0.0', port=5000, debug=True)
