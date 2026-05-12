from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import sqlite3
import os
import sys
import subprocess
from utils.camera import generate_frames, get_auth_status, get_auth_meta, reset_auth_status

app = Flask(__name__)
app.secret_key = "secret123"

DB_PATH = "database/users.db"
FALLBACK_DB_PATH = "database/users_app.db"

# ---------------- DB FUNCTION ----------------
def init_db():
    global DB_PATH
    os.makedirs("database", exist_ok=True)
    target_paths = [DB_PATH, FALLBACK_DB_PATH]

    for path in target_paths:
        try:
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
                """
            )
            conn.commit()
            conn.close()
            DB_PATH = path
            ensure_profile_columns(DB_PATH)
            return
        except sqlite3.DatabaseError:
            # try fallback database path when the default DB is invalid
            continue

    # final guaranteed setup with fallback DB file
    conn = sqlite3.connect(FALLBACK_DB_PATH)
    cur = conn.cursor()
    cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
    )
    conn.commit()
    conn.close()
    DB_PATH = FALLBACK_DB_PATH
    ensure_profile_columns(DB_PATH)


def ensure_profile_columns(db_path):
    """Ensure optional profile fields exist for profile edit feature."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cur.fetchall()]

    if "location" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN location TEXT")
    if "career_goal" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN career_goal TEXT")

    conn.commit()
    conn.close()


def check_user(email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = cur.fetchone()

        conn.close()
        return user
    except sqlite3.DatabaseError:
        return None


def create_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, password),
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Email already exists. Please login instead."
    finally:
        conn.close()


def get_user_profile(email):
    defaults = {
        "name": "",
        "email": email,
        "location": "India",
        "career_goal": "AI Engineer / Software Developer",
    }
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT name, email, location, career_goal FROM users WHERE email=?",
            (email,),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return defaults

        return {
            "name": row[0] or "",
            "email": row[1] or email,
            "location": row[2] or "India",
            "career_goal": row[3] or "AI Engineer / Software Developer",
        }
    except sqlite3.DatabaseError:
        return defaults


def update_user_profile(email, name, location, career_goal):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE users
        SET name=?, location=?, career_goal=?
        WHERE email=?
        """,
        (name.strip(), location.strip(), career_goal.strip(), email),
    )
    conn.commit()
    conn.close()


def train_face_model():
    subprocess.run([sys.executable, "models/encode_faces.py"], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))


init_db()


# ---------------- ROUTES ----------------

@app.route('/')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/register', methods=['POST'])
def register_post():
    from utils.capture_dataset import capture_images

    name = request.form['name'].strip()
    email = request.form['email'].strip()
    password = request.form['password']

    if not name or not email or not password:
        return render_template('register.html', error="All fields are required.")

    success, error = create_user(name, email, password)
    if not success:
        return render_template('register.html', error=error)

    try:
        capture_images(name, target_count=20)
        train_face_model()
    except Exception as ex:
        return render_template(
            'register.html',
            error=f"Account created, but face training failed: {ex}"
        )

    return render_template(
        'register.html',
        message="✅ Registration complete. Face model trained. You can login now."
    )


@app.route('/login', methods=['POST'])
def login_post():
    email = request.form['email']
    password = request.form['password']

    user = check_user(email, password)

    if user:
        session['user'] = email
        session['face_verified'] = False
        return redirect(url_for('face_auth'))
    else:
        return "❌ Invalid Credentials"


@app.route('/face_auth')
def face_auth():
    if 'user' not in session:
        return redirect(url_for('login'))
    reset_auth_status()
    return render_template('face_auth.html')


@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/auth-status')
def auth_status():
    status = get_auth_status()
    meta = get_auth_meta()
    if status == "match":
        session['face_verified'] = True
    return jsonify({"status": status, "meta": meta})


@app.route('/verify_face')
def verify_face():
    # Avoid opening camera twice while /video_feed stream is active.
    if get_auth_status() == "match":
        session['face_verified'] = True
        return redirect(url_for('portfolio'))
    return redirect(url_for('face_auth'))


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    if not session.get('face_verified'):
        return redirect(url_for('face_auth'))

    profile = get_user_profile(session['user'])
    profile_updated = request.args.get("profile_updated") == "1"
    return render_template('dashboard.html', user=session['user'], profile=profile, profile_updated=profile_updated)


@app.route('/portfolio')
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))
    if not session.get('face_verified'):
        return redirect(url_for('face_auth'))

    profile = get_user_profile(session['user'])
    projects = [
        {
            "title": "AI Face Authenticator",
            "stack": "Flask • OpenCV • face_recognition • SQLite",
            "description": "Secure login system with email/password + face verification (2FA), session protection, and user profile management.",
            "status": "Completed",
            "year": "2026",
            "role": "Full Stack + AI Integration",
            "github": "#",
            "demo": "#",
            "highlights": ["2FA Face Login", "Session Security", "Responsive Dashboard"],
        },
        {
            "title": "Student Portfolio Web UI",
            "stack": "HTML • CSS • JavaScript",
            "description": "Responsive portfolio-style interface for profile details, records, and project showcase cards.",
            "status": "In Progress",
            "year": "2026",
            "role": "Frontend Developer",
            "github": "#",
            "demo": "#",
            "highlights": ["Dark/Light Theme", "Card-Based Layout", "Interactive Widgets"],
        },
        {
            "title": "Face Dataset & Encoding Pipeline",
            "stack": "Python • OpenCV • NumPy",
            "description": "Image capture utility and encoding workflow to train/refresh known face embeddings for recognition.",
            "status": "Completed",
            "year": "2025",
            "role": "ML/Computer Vision",
            "github": "#",
            "demo": "#",
            "highlights": ["Dataset Capture", "Encoding Automation", "Model Refresh"],
        },
    ]
    project_stats = {
        "total": len(projects),
        "completed": len([p for p in projects if p.get("status") == "Completed"]),
        "in_progress": len([p for p in projects if p.get("status") == "In Progress"]),
    }
    return render_template('portfolio.html', user=session['user'], profile=profile, projects=projects, project_stats=project_stats)


@app.route('/profile/update', methods=['POST'])
def profile_update():
    if 'user' not in session:
        return redirect(url_for('login'))
    if not session.get('face_verified'):
        return redirect(url_for('face_auth'))

    name = request.form.get('name', '').strip()
    location = request.form.get('location', '').strip()
    career_goal = request.form.get('career_goal', '').strip()

    if not name:
        return redirect(url_for('dashboard'))

    update_user_profile(session['user'], name, location, career_goal)
    return redirect(url_for('dashboard', profile_updated='1'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)