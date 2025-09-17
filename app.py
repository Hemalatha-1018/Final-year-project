from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pymysql
import hashlib
import re
import requests
import logging
import json
from datetime import datetime
from functools import wraps
from uuid import uuid4
from html import escape
import time
from models.Aiargument import *

recent_queries_cache = {}
CACHE_EXPIRY_SECONDS = 30

app = Flask(__name__)
app.secret_key = 'e6e989e428fdb523f19d73132267f7a75e540b524df85852db2ea30d355731a7'

db_config = {
    "host": "mysql-11ce5b3-rubikproxy.g.aivencloud.com",
    "port": 13817,
    "user": "avnadmin",
    "password": "AVNS_GA8NNQA-ZaPnNLbVpwa",
    "db": "defaultdb",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

SCHEME_PROMPT = response

schemebot_model_key = ai_model_key
schemebot_model_link = ai_model_url

request_counts = {}
RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60


def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user', {}).get('id', 'guest')
        current_time = datetime.now().timestamp()
        
        request_counts.setdefault(user_id, [])
        request_counts[user_id] = [t for t in request_counts[user_id] if current_time - t < RATE_LIMIT_WINDOW]
        
        if len(request_counts[user_id]) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        request_counts[user_id].append(current_time)
        return f(*args, **kwargs)
    return decorated_function

def verify_user_exists(user_id):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone() is not None
    except pymysql.Error as e:
        print(f"Error verifying user: {str(e)}")
        return False
    finally:
        if 'connection' in locals() and connection:
            connection.close()

def get_db_connection():
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            connection = pymysql.connect(**db_config)
            print("Database connection successful")
            return connection
        except pymysql.Error as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)
            retry_delay *= 2

def initialize_database():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    email VARCHAR(100) NOT NULL UNIQUE,
                    fullname VARCHAR(100),
                    phone VARCHAR(20),
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferred_language VARCHAR(10) DEFAULT 'English',
                    INDEX idx_username (username),
                    INDEX idx_email (email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    message TEXT NOT NULL,
                    is_user BOOLEAN NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_id (user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            
            connection.commit()
            print("Database tables initialized")
    except pymysql.Error as e:
        print(f"Database initialization failed: {str(e)}")
        raise
    finally:
        if 'connection' in locals() and connection:
            connection.close()

try:
    initialize_database()
except Exception as e:
    print(f"Database initialization error: {str(e)}")

def get_chat_context(user_id, limit=10):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT message, is_user 
                FROM chat_history 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s""", (user_id, limit))
            return cursor.fetchall()
    except pymysql.Error as e:
        print(f"Error fetching chat context: {str(e)}")
        return []
    finally:
        if 'connection' in locals() and connection:
            connection.close()

def save_chat_message(user_id, message, is_user):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO chat_history (user_id, message, is_user) 
                VALUES (%s, %s, %s)""", 
                (user_id, message, is_user))
            connection.commit()
    except pymysql.Error as e:
        print(f"Error saving chat message: {str(e)}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()

def generate_quick_replies(response_text, user_id):
    try:
        suggestions = [
            "More details", 
            "Eligibility criteria",
            "How to apply",
            "Official website"
        ]
        
        response_lower = response_text.lower()
        
        if "scheme" in response_lower or "program" in response_lower:
            suggestions.extend([
                "Similar schemes",
                "Benefits overview",
                "Required documents"
            ])
        
        if "eligibility" in response_lower:
            suggestions.extend([
                "Check my eligibility",
                "Required income level",
                "Age requirements"
            ])
        
        unique_suggestions = list(dict.fromkeys(suggestions))
        
        return unique_suggestions[:6]
    except Exception as e:
        print(f"Error generating quick replies: {str(e)}")
        return [
            "More details",
            "Eligibility",
            "How to apply",
            "Official website"
        ]

def get_user_context(user_id):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT fullname, email, phone, preferred_language 
                FROM users 
                WHERE id = %s""", (user_id,))
            user = cursor.fetchone()
            if user:
                return {
                    "fullname": user["fullname"] or "Unknown",
                    "email": user["email"],
                    "phone": user["phone"] or "Unknown",
                    "preferred_language": user["preferred_language"]
                }
            return {}
    except pymysql.Error as e:
        print(f"Error fetching user context: {str(e)}")
        return {}
    finally:
        if 'connection' in locals() and connection:
            connection.close()

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/login', methods=['GET'])
def login_page():
    return render_template("login.html")

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT message, is_user, timestamp 
                FROM chat_history 
                WHERE user_id=%s 
                ORDER BY timestamp""", 
                (session['user']['id'],))
            chat_history = cursor.fetchall()
        
        return render_template("chat.html", 
                            chat_history=chat_history)
    except pymysql.Error as e:
        print(f"Database error loading chat history: {str(e)}")
        return render_template("chat.html", 
                            chat_history=[])
    except Exception as e:
        print(f"Error loading chat: {str(e)}")
        return render_template("chat.html", 
                            chat_history=[])
    finally:
        if 'connection' in locals() and connection:
            connection.close()

@app.route('/complaint' , methods=['GET'])
def complaint():
    return render_template("complaint.html")

@app.route('/get_schemes', methods=['POST'])
@rate_limit
def get_schemes():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not verify_user_exists(session['user']['id']):
        session.pop('user', None)
        return jsonify({"error": "User session invalid"}), 401

    try:
        data = request.get_json()
        user_query = escape(data.get('query', '').strip())
        user_id = session['user']['id']

        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        cache_key = f"{user_id}:{user_query}"
        current_time = time.time()

        cached = recent_queries_cache.get(cache_key)
        if cached and current_time - cached[1] < CACHE_EXPIRY_SECONDS:
            return jsonify(cached[0]), 200

        user_context = get_user_context(user_id)
        context_str = json.dumps(user_context)

        chat_context = get_chat_context(user_id, limit=10)
        context_messages = [
            {"role": "user" if msg['is_user'] else "assistant", "content": msg['message']}
            for msg in reversed(chat_context)
        ]

        prompt = SCHEME_PROMPT.replace("{query}", user_query).replace("{context}", context_str)
        messages = [{"role": "system", "content": prompt}] + context_messages + [{"role": "user", "content": user_query}]
        headers = {
            "Authorization": f"Bearer {schemebot_model_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        response = requests.post(schemebot_model_link, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        assistant_response = response_data['choices'][0]['message']['content']
        quick_replies = generate_quick_replies(assistant_response, user_id)

        save_chat_message(user_id, user_query, True)
        save_chat_message(user_id, assistant_response, False)

        final_response = {
            "response": assistant_response,
            "quick_replies": quick_replies,
            "success": True
        }


        recent_queries_cache[cache_key] = (final_response, current_time)

        return jsonify(final_response), 200

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return jsonify({"error": "Rate limit exceeded. Please wait and try again."}), 429
        print(f"API request failed: {str(e)}")
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except KeyError as e:
        print(f"Invalid response format: {str(e)}")
        return jsonify({"error": "Invalid response format from API"}), 500
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/clear_chat', methods=['POST'])
@rate_limit
def clear_chat():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not verify_user_exists(session['user']['id']):
        session.pop('user', None)
        return jsonify({"error": "User session invalid"}), 401

    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chat_history WHERE user_id = %s", (session['user']['id'],))
            connection.commit()
        return jsonify({"success": True, "message": "Chat history cleared"}), 200
    except pymysql.Error as e:
        print(f"Database error clearing chat history: {str(e)}")
        return jsonify({"error": "Failed to clear chat history"}), 500
    finally:
        if 'connection' in locals() and connection:
            connection.close()

@app.route('/register', methods=['POST'])
def register():
    try:
        username = request.form['username']
        email = request.form['email']
        fullname = request.form['fullname']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validate inputs
        if len(username) < 4:
            return jsonify({"error": "Username must be at least 4 characters"}), 400
            
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return jsonify({"error": "Invalid email format"}), 400
            
        if not re.match(r'^[0-9]{10,15}$', phone):
            return jsonify({"error": "Invalid phone number"}), 400
            
        if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'[a-z]', password) or not re.search(r'[0-9]', password):
            return jsonify({"error": "Password must be at least 8 characters with uppercase, lowercase and number"}), 400
            
        if password != confirm_password:
            return jsonify({"error": "Passwords do not match"}), 400

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, email, fullname, phone, password) VALUES (%s, %s, %s, %s, %s)",
                (username, email, fullname, phone, password_hash)
            )
            connection.commit()
            return jsonify({"success": True}), 200
    except pymysql.err.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 400
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500
    finally:
        if 'connection' in locals() and connection:
            connection.close()

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()

        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            user = cursor.fetchone()

            if user:
                session['user'] = user
                return jsonify({"success": True}), 200
            else:
                return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500
    finally:
        if 'connection' in locals() and connection:
            connection.close()

def sanitize_input(text):
    return escape(text)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        try:
            data = request.form
            fullname = sanitize_input(data.get('fullname', ''))
            email = sanitize_input(data.get('email', ''))
            phone = sanitize_input(data.get('phone', ''))
            preferred_language = sanitize_input(data.get('preferred_language', 'English'))

            if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
                return jsonify({"error": "Invalid email format"}), 400
            if phone and not re.match(r'^[0-9]{10,15}$', phone):
                return jsonify({"error": "Invalid phone number"}), 400

            connection = get_db_connection()
            with connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE users 
                    SET fullname=%s, email=%s, phone=%s, preferred_language=%s
                    WHERE id=%s""", 
                    (fullname, email, phone, preferred_language, session['user']['id']))
                
                connection.commit()
                
                cursor.execute("SELECT * FROM users WHERE id=%s", (session['user']['id'],))
                session['user'] = cursor.fetchone()
                
                return jsonify({
                    "success": True, 
                    "message": "Profile updated"
                }), 200
        except pymysql.err.IntegrityError:
            return jsonify({"error": "Email already in use"}), 400
        except pymysql.Error as e:
            print(f"Database error updating profile: {str(e)}")
            return jsonify({"error": "Database operation failed"}), 500
        except Exception as e:
            print(f"Error updating profile: {str(e)}")
            return jsonify({"error": "Profile update failed"}), 500
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    return render_template("profile.html", 
                        user=session['user'])

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))
