# --- IMPORTS ---
import sqlite3
import re
import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import docx2txt
from PyPDF2 import PdfReader
import torch
from transformers import pipeline
import random
import plotly.express as px

# --- SQLite SETUP ---
# Connect SQLite database file for storing flashcards and quiz data
DB_PATH = "learning.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# --- DATABASE INITIALIZATION ---

def initialize_database():
    """Initialize or reset the database tables"""
    cursor.execute("""CREATE TABLE IF NOT EXISTS flashcards 
                   (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   question TEXT,
                   correct_answer TEXT,
                   wrong1 TEXT,
                   wrong2 TEXT,
                   difficulty TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   last_reviewed TIMESTAMP,
                   review_count INTEGER DEFAULT 0,
                   leitner_box INTEGER DEFAULT 0,  -- Add Leitner box (0-5)
                   next_review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- When to review next
                   )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS quiz_sessions 
                   (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   session_name TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   completed BOOLEAN DEFAULT FALSE,
                   score INTEGER
                   )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS quiz_questions 
                   (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   session_id INTEGER,
                   flashcard_id INTEGER,
                   user_answer TEXT,
                   is_correct BOOLEAN,
                   FOREIGN KEY(session_id) REFERENCES quiz_sessions(id),
                   FOREIGN KEY(flashcard_id) REFERENCES flashcards(id)
                   )""")
    conn.commit()
    
def upgrade_database():
    """Upgrade existing database to add new columns if they don't exist"""
    try:
        # Check if leitner_box column exists
        cursor.execute("PRAGMA table_info(flashcards)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'leitner_box' not in columns:
            cursor.execute("ALTER TABLE flashcards ADD COLUMN leitner_box INTEGER DEFAULT 0")
            st.sidebar.info("Added leitner_box column to database")
        
        if 'next_review_date' not in columns:
            cursor.execute("ALTER TABLE flashcards ADD COLUMN next_review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            st.sidebar.info("Added next_review_date column to database")
            
        conn.commit()
    except Exception as e:
        st.sidebar.error(f"Error upgrading database: {e}")
# Initialize database tables
initialize_database()
upgrade_database()

# --- TEXT PROCESSING FUNCTIONS ---
def extract_text(file):
    """Extracts text from uploaded files (PDF, DOCX, or TXT)"""
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/msword"]:
        text = docx2txt.process(file)
    elif file.type.startswith("text"):
        stringio = file.getvalue().decode("utf-8", errors="ignore")
        text = stringio
    else:
        st.warning("Unsupported file type")
    return text

def clean_text(text):
    """Cleans and normalizes extracted text"""
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

# --- QA GENERATION FUNCTIONS ---
@st.cache_resource
def load_qa_model():
    """Loads and caches the Hugging Face question generation model"""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=device
    )

qa_pipeline = load_qa_model()  

def generate_qa(text: str, num_questions: int = 5) -> List[List[str]]:
    """Generates quiz questions from text"""
    qa_pairs = []
    attempts = 0
    max_attempts = num_questions * 3
    
    chunk_size = 1000
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] if len(text) > chunk_size else [text]
    
    while len(qa_pairs) < num_questions and attempts < max_attempts:
        attempts += 1
        
        try:
            context_chunk = random.choice(text_chunks) if text_chunks else text
            
            prompt_templates = [
                f"Generate one multiple-choice question with 3 options and mark the correct answer from: {context_chunk}",
                f"Create a quiz question with 4 choices (1 correct, 3 wrong) from this text: {context_chunk}",
                f"From the following text, make a MCQ with one correct and two wrong answers: {context_chunk}\nFormat: Question? A) Option1 B) Option2 C) Option3 (Answer: A)"
            ]
            
            prompt = random.choice(prompt_templates)

            result = qa_pipeline(
                prompt,
                max_length=512,
                do_sample=True,
                temperature=0.85,
                top_k=50,
                num_return_sequences=1
            )
            
            qa_text = result[0]['generated_text'].strip()
            question_data = parse_any_format(qa_text)
            
            if question_data and not any(q[0] == question_data[0] for q in qa_pairs):
                qa_pairs.append(question_data)
                
        except Exception as e:
            st.warning(f"Error generating question: {str(e)}")
            continue
            
    if not qa_pairs and attempts >= max_attempts:
        st.error("Failed to generate flashcards after multiple attempts. The text might be too short or complex.")
    
    return qa_pairs[:num_questions]

def parse_any_format(qa_text: str) -> Optional[List[str]]:
    """Parses generated questions into standardized format"""
    qa_text = qa_text.replace("\n", " ").strip()
    
    question_match = re.search(r'^(.*?\?)', qa_text)
    if not question_match:
        return None
    
    question = question_match.group(1).strip()
    
    option_patterns = [
        r'[A-D][):.]?\s*(.*?)(?:\s*(?:[A-D][):.]|$))',
        r'\d[.:]\s*(.*?)(?:\s*(?:\d[.:]|$))',
        r'-\s*(.*?)(?:\s*(?:-|$))',
        r'option\s*\w\s*:\s*(.*?)(?:\s*(?:option\s*\w\s*:|$))'
    ]
    
    options = []
    for pattern in option_patterns:
        options = re.findall(pattern, qa_text, re.IGNORECASE)
        if len(options) >= 3:
            break
    
    if len(options) < 3:
        remaining_text = qa_text[len(question):].strip()
        if remaining_text:
            options = remaining_text.split()[:3]
        else:
            options = ["Correct answer", "Wrong option 1", "Wrong option 2"]
    
    answer_patterns = [
        r'answer\s*[:\-]\s*([A-D1-3])',
        r'correct\s*[:\-]\s*([A-D1-3])',
        r'right\s*[:\-]\s*([A-D1-3])'
    ]
    
    correct_idx = 0
    for pattern in answer_patterns:
        answer_match = re.search(pattern, qa_text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            if answer.isdigit():
                correct_idx = int(answer) - 1
            else:
                correct_idx = ord(answer) - ord('A')
            correct_idx = max(0, min(correct_idx, len(options)-1))
            break
    
    while len(options) < 3:
        options.append(f"Option {len(options)+1}")
    
    correct = options[correct_idx]
    wrongs = [opt for i, opt in enumerate(options) if i != correct_idx]
    
    while len(wrongs) < 2:
        wrongs.append(f"Wrong option {len(wrongs)+1}")
    
    return [question, correct, wrongs[0], wrongs[1]]

# --- LEITNER ALGORITHM FUNCTIONS ---
def update_leitner_box(card_id, is_correct):
    """Update the Leitner box based on whether the answer was correct"""
    try:
        cursor.execute("SELECT leitner_box FROM flashcards WHERE id=?", (card_id,))
        result = cursor.fetchone()
        current_box = result[0] if result else 0
        
        if is_correct:
            new_box = min(current_box + 1, 5)  # Move to next box, max is box 5
        else:
            new_box = max(current_box - 1, 0)  # Move back a box, min is box 0
        
        # Calculate next review date based on box number
        intervals = [1, 2, 7, 14, 30, 90]  # Days until next review
        next_review = datetime.now() + timedelta(days=intervals[new_box])
        
        cursor.execute(
            """UPDATE flashcards 
            SET leitner_box=?, next_review_date=?, last_reviewed=CURRENT_TIMESTAMP, review_count=review_count+1 
            WHERE id=?""",
            (new_box, next_review, card_id)
        )
        conn.commit()
        return new_box
    except sqlite3.OperationalError:
        # If columns don't exist yet, just update the basic fields
        cursor.execute(
            """UPDATE flashcards 
            SET last_reviewed=CURRENT_TIMESTAMP, review_count=review_count+1 
            WHERE id=?""",
            (card_id,)
        )
        conn.commit()
        return 0

def get_due_flashcards():
    """Get flashcards that are due for review based on Leitner system"""
    query = """SELECT * FROM flashcards 
               WHERE next_review_date <= datetime('now') 
               ORDER BY leitner_box ASC, next_review_date ASC"""
    df = pd.read_sql_query(query, conn)
    return df

def get_leitner_stats():
    """Get statistics about Leitner boxes"""
    try:
        # Check if leitner_box column exists
        cursor.execute("PRAGMA table_info(flashcards)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'leitner_box' not in columns:
            return pd.DataFrame(columns=['leitner_box', 'count'])
            
        query = """SELECT leitner_box, COUNT(*) as count 
                   FROM flashcards 
                   GROUP BY leitner_box 
                   ORDER BY leitner_box"""
        df = pd.read_sql_query(query, conn)
        return df
    except:
        # If any error occurs, return empty DataFrame
        return pd.DataFrame(columns=['leitner_box', 'count'])
# --- DATABASE OPERATIONS ---
def save_flashcard(question, correct, wrong1, wrong2, difficulty):
    """Saves a new flashcard to database"""
    cursor.execute(
        """INSERT INTO flashcards 
        (question, correct_answer, wrong1, wrong2, difficulty) 
        VALUES (?, ?, ?, ?, ?)""",
        (question, correct, wrong1, wrong2, difficulty)
    )
    conn.commit()

def get_flashcards(filter_difficulty=None):
    """Retrieves flashcards with optional difficulty filter"""
    query = "SELECT * FROM flashcards"
    params = ()
    
    if filter_difficulty:
        query += " WHERE difficulty = ?"
        params = (filter_difficulty,)
        
    query += " ORDER BY last_reviewed ASC, review_count ASC"
    df = pd.read_sql_query(query, conn, params=params)
    return df

def update_flashcard(card_id, question, correct, wrong1, wrong2, difficulty):
    """Updates an existing flashcard"""
    cursor.execute(
        """UPDATE flashcards 
        SET question=?, correct_answer=?, wrong1=?, wrong2=?, difficulty=?, last_reviewed=CURRENT_TIMESTAMP 
        WHERE id=?""",
        (question, correct, wrong1, wrong2, difficulty, card_id)
    )
    conn.commit()

def delete_flashcard(card_id):
    """Deletes a flashcard from database"""
    cursor.execute("DELETE FROM flashcards WHERE id=?", (card_id,))
    conn.commit()

def create_quiz_session(session_name):
    """Creates a new quiz session"""
    cursor.execute(
        "INSERT INTO quiz_sessions (session_name) VALUES (?)",
        (session_name,)
    )
    conn.commit()
    return cursor.lastrowid

def add_question_to_quiz(session_id, flashcard_id):
    """Adds a question to a quiz session"""
    cursor.execute(
        """INSERT INTO quiz_questions (session_id, flashcard_id) 
        VALUES (?, ?)""",
        (session_id, flashcard_id)
    )
    conn.commit()

def record_quiz_answer(question_id, user_answer, is_correct, flashcard_id):
    """Records a user's answer to a quiz question and updates Leitner box"""
    cursor.execute(
        """UPDATE quiz_questions 
        SET user_answer=?, is_correct=? 
        WHERE id=?""",
        (user_answer, is_correct, question_id)
    )
    
    # Update the Leitner box based on correctness
    update_leitner_box(flashcard_id, is_correct)
    conn.commit()

def complete_quiz_session(session_id, score):
    """Marks a quiz session as completed with score"""
    cursor.execute(
        """UPDATE quiz_sessions 
        SET completed=TRUE, score=? 
        WHERE id=?""",
        (score, session_id)
    )
    conn.commit()

def get_quiz_session(session_id):
    """Gets quiz session details"""
    cursor.execute("SELECT * FROM quiz_sessions WHERE id=?", (session_id,))
    return cursor.fetchone()

def get_quiz_questions(session_id):
    """Gets all questions for a quiz session"""
    df = pd.read_sql_query(
        """SELECT q.id, q.session_id, q.flashcard_id, q.user_answer, q.is_correct,
           f.question, f.correct_answer, f.wrong1, f.wrong2, f.difficulty
           FROM quiz_questions q
           JOIN flashcards f ON q.flashcard_id = f.id
           WHERE q.session_id = ?""",
        conn, params=(session_id,)
    )
    return df

def get_quiz_history():
    """Gets history of completed quizzes"""
    df = pd.read_sql_query(
        "SELECT id, session_name, created_at, score FROM quiz_sessions WHERE completed=TRUE ORDER BY created_at DESC",
        conn
    )
    return df

# --- QUIZ FUNCTIONS ---
def generate_quiz_questions(difficulty=None, limit=10):
    """Selects questions for a new quiz"""
    query = "SELECT * FROM flashcards"
    params = ()
    
    if difficulty:
        query += " WHERE difficulty = ?"
        params = (difficulty,)
        
    query += " ORDER BY RANDOM() LIMIT ?"
    params += (limit,)
    
    df = pd.read_sql_query(query, conn, params=params)
    return df.to_dict('records')

def calculate_quiz_score(session_id):
    """Calculates score for a quiz session"""
    cursor.execute(
        "SELECT COUNT(*) FROM quiz_questions WHERE session_id=? AND is_correct=TRUE",
        (session_id,)
    )
    correct = cursor.fetchone()[0]
    
    cursor.execute(
        "SELECT COUNT(*) FROM quiz_questions WHERE session_id=?",
        (session_id,)
    )
    total = cursor.fetchone()[0]
    
    return int((correct / total) * 100) if total > 0 else 0

# --- STREAMLIT PAGES ---
def generate_flashcards_page():
    """Page for generating flashcards from documents"""
    st.title("📄 Flashcard Generator")
    st.info("Upload a document, choose how many questions you want, and then generate flashcards. Pick the ones you want to keep!")

    # Debugging and reset options
    with st.sidebar.expander("Developer Tools"):
        if st.button("🔄 Clear All Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("All caches cleared!")
        
        if st.button("💣 Reset Entire Database"):
            initialize_database()
            st.success("Database reset complete!")

    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, or TXT)", 
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, Word DOCX, or plain text"
    )
    
    if uploaded_file is not None:
        extracted_text = extract_text(uploaded_file)
        cleaned = clean_text(extracted_text) if extracted_text else ""
        
        view_option = st.radio(
            "Select view:",
            ("Show Extracted Text", "Show Cleaned Text"),
            horizontal=True
        )
        
        if view_option == "Show Extracted Text":
            st.subheader("📜 Extracted Text")
            st.text(extracted_text)
            st.download_button(label="Download Extracted Text",
                             data=extracted_text,
                             file_name="extracted_text.txt",
                             mime="text/plain")
        else:
            st.subheader("✨ Cleaned Text")
            st.text(cleaned)
            st.download_button(label="Download Cleaned Text",
                             data=cleaned,
                             file_name="cleaned_text.txt",
                             mime="text/plain")            
    
        num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=20, value=5, step=1)
    
        if st.button("Generate Quiz"):
            if not cleaned:
                st.warning("No text available to generate questions")
            else:
                with st.spinner("Generating flashcards..."):
                    qa_pairs = generate_qa(cleaned, num_questions=num_questions)
                    if qa_pairs:
                        st.session_state.qa_pairs = qa_pairs
                        st.success(f"Successfully generated {len(qa_pairs)} flashcard(s)!")
                    else:
                        st.error("Failed to generate flashcards. Try with different text.")
    
    if "qa_pairs" in st.session_state:
        st.subheader("📝 Generated Flashcards")
        flashcards_to_save = []
        with st.form("save_flashcards_form"):
            for i, (question, correct, wrong1, wrong2) in enumerate(st.session_state.qa_pairs, 1):
                with st.expander(f"Flashcard {i}: {question}", expanded=False):
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Correct Answer:** {correct}")
                    st.markdown(f"**Option 1:** {wrong1}")
                    st.markdown(f"**Option 2:** {wrong2}")
                    difficulty = st.selectbox(f"Select difficulty", 
                                            ["Easy", "Medium", "Hard"], 
                                            key=f"diff_{i}")
                    save_option = st.checkbox("Keep this flashcard?", key=f"keep_{i}")
                    
                    if save_option:
                        flashcards_to_save.append({
                            "question": question,
                            "correct": correct,
                            "wrong1": wrong1,
                            "wrong2": wrong2,
                            "difficulty": difficulty
                        })
            submitted = st.form_submit_button("Save Selected Flashcards")
            if submitted:
                if flashcards_to_save:
                    for card in flashcards_to_save:
                        save_flashcard(card["question"], card["correct"], card["wrong1"], card["wrong2"], card["difficulty"])
                    st.success(f"Saved {len(flashcards_to_save)} flashcards!")
                    st.balloons()
                    del st.session_state.qa_pairs
                    st.rerun()
                else:
                    st.info("No flashcards selected to save.")

def manage_flashcards_page():
    """Page for managing existing flashcards"""
    st.title("🗄️ Flashcards Management")
    st.info("Edit, delete, or organize your saved flashcards below.")
    
    # Debugging and reset options
    with st.sidebar.expander("Developer Tools"):
        if st.button("🔄 Clear All Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("All caches cleared!")
        
        if st.button("💣 Reset Entire Database"):
            initialize_database()
            st.success("Database reset complete!")
    
    # Add filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_difficulty = st.selectbox(
            "Filter by difficulty",
            ["All", "Easy", "Medium", "Hard"],
            index=0
        )
    
    with col2:
        sort_option = st.selectbox(
            "Sort by",
            ["Recently Added", "Least Reviewed", "Difficulty"],
            index=0
        )
    
    # Get filtered flashcards
    difficulty = filter_difficulty if filter_difficulty != "All" else None
    df = get_flashcards(difficulty)
    
    if df.empty:
        st.warning("No flashcards found. Generate some first!")
    else:
        st.write(f"Found {len(df)} flashcards")
        
        # Display Leitner stats
        leitner_stats = get_leitner_stats()
        if not leitner_stats.empty and not leitner_stats.isna().all().all():
            st.subheader("Leitner System Progress")
            col1, col2, col3 = st.columns(3)
            
            total_cards = leitner_stats['count'].sum()
            mastered_cards = leitner_stats[leitner_stats['leitner_box'] == 5]['count'].sum() if 5 in leitner_stats['leitner_box'].values else 0
            
            with col1:
                st.metric("Total Cards", total_cards)
            with col2:
                st.metric("Mastered Cards", mastered_cards)
            with col3:
                if total_cards > 0:
                    st.metric("Mastery Rate", f"{(mastered_cards/total_cards)*100:.1f}%")
            
            # Show box distribution
            fig = px.bar(leitner_stats, x='leitner_box', y='count', 
                         title="Cards in Each Leitner Box",
                         labels={'leitner_box': 'Leitner Box', 'count': 'Number of Cards'})
            st.plotly_chart(fig)
        else:
            st.info("Leitner system statistics will appear after you start using the study system.")
        
        # Display flashcards in individual expanders with edit options
        for index, row in df.iterrows():
            with st.expander(f"Card #{row['id']} ({row['difficulty']}): {row['question']}", expanded=False):
                with st.form(key=f"edit_form_{row['id']}"):
                    new_question = st.text_area("Question", row["question"], key=f"question_{row['id']}")
                    new_correct = st.text_input("Correct Answer", row["correct_answer"], key=f"correct_{row['id']}")
                    new_wrong1 = st.text_input("Wrong Option 1", row["wrong1"], key=f"wrong1_{row['id']}")
                    new_wrong2 = st.text_input("Wrong Option 2", row["wrong2"], key=f"wrong2_{row['id']}")
                    new_diff = st.selectbox(
                        "Difficulty", 
                        ["Easy", "Medium", "Hard"], 
                        index=["Easy", "Medium", "Hard"].index(row["difficulty"]), 
                        key=f"diff_{row['id']}"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        update_btn = st.form_submit_button("Update")
                    with col2:
                        delete_btn = st.form_submit_button("Delete")
                    with col3:
                        quiz_btn = st.form_submit_button("Add to Quiz")
                    
                    if update_btn:
                        update_flashcard(row['id'], new_question, new_correct, new_wrong1, new_wrong2, new_diff)
                        st.success("Flashcard updated!")
                        st.rerun()
                    if delete_btn:
                        delete_flashcard(row['id'])
                        st.success("Flashcard deleted!")
                        st.rerun()
                    if quiz_btn:
                        if "quiz_flashcards" not in st.session_state:
                            st.session_state.quiz_flashcards = []
                        st.session_state.quiz_flashcards.append(row['id'])
                        st.success("Added to quiz selection!")

def leitner_study_page():
    """Page for studying with the Leitner algorithm"""
    st.title("📊 Leitner Study System")
    st.info("Study your flashcards using the spaced repetition method. Cards you know well appear less often.")
    
    # Debugging and reset options
    with st.sidebar.expander("Developer Tools"):
        if st.button("🔄 Clear All Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("All caches cleared!")
        
        if st.button("💣 Reset Entire Database"):
            initialize_database()
            st.success("Database reset complete!")
    
    # Show Leitner statistics
    stats_df = get_leitner_stats()
    if not stats_df.empty and not stats_df.isna().all().all():
        st.subheader("Your Progress")
        col1, col2, col3 = st.columns(3)
        
        total_cards = stats_df['count'].sum()
        mastered_cards = stats_df[stats_df['leitner_box'] == 5]['count'].sum() if 5 in stats_df['leitner_box'].values else 0
        
        with col1:
            st.metric("Total Cards", total_cards)
        with col2:
            st.metric("Mastered Cards", mastered_cards)
        with col3:
            if total_cards > 0:
                st.metric("Mastery Rate", f"{(mastered_cards/total_cards)*100:.1f}%")
        
        # Show box distribution
        fig = px.bar(stats_df, x='leitner_box', y='count', 
                     title="Cards in Each Leitner Box",
                     labels={'leitner_box': 'Leitner Box', 'count': 'Number of Cards'})
        st.plotly_chart(fig)
    else:
        st.info("Leitner system statistics will appear after you start studying.")
    
    # Get due flashcards
    due_cards = get_due_flashcards()
    
    if due_cards.empty:
        st.success("🎉 No flashcards due for review! You're all caught up.")
        return
    
    st.subheader(f"Due for Review: {len(due_cards)} cards")
    
    # Study session
    if "current_study_index" not in st.session_state:
        st.session_state.current_study_index = 0
        st.session_state.study_cards = due_cards.to_dict('records')
        st.session_state.show_answer = False
    
    current_index = st.session_state.current_study_index
    current_card = st.session_state.study_cards[current_index]
    
    # Display progress
    progress = (current_index + 1) / len(st.session_state.study_cards)
    st.progress(progress)
    st.caption(f"Card {current_index + 1} of {len(st.session_state.study_cards)}")
    
    # Display question
    st.subheader("Question")
    st.write(current_card['question'])
    
    if not st.session_state.show_answer:
        if st.button("Show Answer"):
            st.session_state.show_answer = True
            st.rerun()
    else:
        st.subheader("Answer")
        st.success(current_card['correct_answer'])
        
        st.subheader("How did you do?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ I knew it", use_container_width=True):
                update_leitner_box(current_card['id'], True)
                next_card()
        with col2:
            if st.button("❌ I didn't know", use_container_width=True):
                update_leitner_box(current_card['id'], False)
                next_card()

def next_card():
    """Move to the next card in the study session"""
    st.session_state.show_answer = False
    if st.session_state.current_study_index < len(st.session_state.study_cards) - 1:
        st.session_state.current_study_index += 1
    else:
        st.session_state.current_study_index = 0  # Restart or could end session
    st.rerun()

def quiz_page():
    """Page for taking and reviewing quizzes"""
    st.title("🎯 Quiz Mode")
    
    # Debugging and reset options
    with st.sidebar.expander("Developer Tools"):
        if st.button("🔄 Clear All Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.success("All caches cleared!")
        
        if st.button("💣 Reset Entire Database"):
            initialize_database()
            st.success("Database reset complete!")
    
    # Initialize session state for quiz if not exists
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = {
            "session_id": None,
            "questions": [],
            "current_question": 0,
            "answers": {},
            "completed": False
        }
    
    # Quiz creation section
    if st.session_state.current_quiz["session_id"] is None:
        st.info("Create a new quiz or review past quiz attempts.")
        
        tab1, tab2 = st.tabs(["New Quiz", "Quiz History"])
        
        with tab1:
            st.subheader("Create New Quiz")
            
            # Quiz configuration
            col1, col2 = st.columns(2)
            with col1:
                quiz_name = st.text_input("Quiz Name", "My Quiz")
                question_count = st.slider("Number of Questions", 5, 20, 10)
            
            with col2:
                difficulty = st.selectbox(
                    "Difficulty Level",
                    ["All", "Easy", "Medium", "Hard"],
                    index=0
                )
                include_selected = st.checkbox("Include selected flashcards", value=True)
            
            if st.button("Start Quiz"):
                # Create a new quiz session
                session_id = create_quiz_session(quiz_name)
                
                # Get questions for the quiz
                selected_questions = []
                
                # Add manually selected flashcards if any
                if include_selected and "quiz_flashcards" in st.session_state:
                    for card_id in st.session_state.quiz_flashcards:
                        add_question_to_quiz(session_id, card_id)
                    selected_questions.extend(st.session_state.quiz_flashcards)
                    del st.session_state.quiz_flashcards
                
                # Get remaining questions randomly
                remaining = question_count - len(selected_questions)
                if remaining > 0:
                    diff = difficulty if difficulty != "All" else None
                    questions = generate_quiz_questions(diff, remaining)
                    for q in questions:
                        add_question_to_quiz(session_id, q['id'])
                
                # Initialize the quiz in session state
                questions_df = get_quiz_questions(session_id)
                st.session_state.current_quiz = {
                    "session_id": session_id,
                    "questions": questions_df.to_dict('records'),
                    "current_question": 0,
                    "answers": {},
                    "completed": False
                }
                
                st.rerun()
        
        with tab2:
            st.subheader("Quiz History")
            history_df = get_quiz_history()
            
            if history_df.empty:
                st.info("No quiz history yet.")
            else:
                for _, row in history_df.iterrows():
                    with st.expander(f"{row['session_name']} - Score: {row['score']}% - {row['created_at']}"):
                        st.write(f"Date: {row['created_at']}")
                        st.write(f"Score: {row['score']}%")
                        if st.button(f"Review Quiz #{row['id']}"):
                            # Load the quiz for review
                            questions_df = get_quiz_questions(row['id'])
                            st.session_state.current_quiz = {
                                "session_id": row['id'],
                                "questions": questions_df.to_dict('records'),
                                "current_question": 0,
                                "answers": {q['id']: q['user_answer'] for _, q in questions_df.iterrows()},
                                "completed": True
                            }
                            st.rerun()
    
    # Quiz taking/review section
    else:
        quiz = st.session_state.current_quiz
        questions = quiz["questions"]
        current_idx = quiz["current_question"]
        question_data = questions[current_idx]
        
        # Clean question text by removing any existing answer prefixes
        clean_question = re.sub(r'[A-Z]\)\s*', '', question_data['question']).strip()
        
        # Display progress
        st.progress((current_idx + 1) / len(questions))
        st.caption(f"Question {current_idx + 1} of {len(questions)}")
        
        # Display the cleaned question
        st.subheader(clean_question)
        
        # Prepare answer options with identifiers
        options = [
            ("A", question_data['correct_answer'].strip()),
            ("B", question_data['wrong1'].strip()),
            ("C", question_data['wrong2'].strip())
        ]
        random.shuffle(options)

        # Find correct answer key
        correct_answer_key = next(key for key, val in options if val == question_data['correct_answer'].strip())

        # If reviewing a completed quiz
        if quiz["completed"]:
            user_answer_key = quiz["answers"].get(question_data['id'])
            was_correct = user_answer_key == correct_answer_key
            
            # Display user's answer and correct answer
            st.markdown(f"**Your answer:** {user_answer_key}) {next(val for k, val in options if k == user_answer_key)}")
            st.markdown(f"**Correct answer:** {correct_answer_key}) {question_data['correct_answer']}")
            
            if was_correct:
                st.success("✅ You got this right!")
            else:
                st.error("❌ You missed this one.")
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("⏮️ Previous") and current_idx > 0:
                    st.session_state.current_quiz["current_question"] -= 1
                    st.rerun()
            with col2:
                if st.button("⏭️ Next") and current_idx < len(questions) - 1:
                    st.session_state.current_quiz["current_question"] += 1
                    st.rerun()
            with col3:
                if st.button("🏁 Finish Review"):
                    del st.session_state.current_quiz
                    st.rerun()
        
        # If taking the quiz
        else:
            selected = st.radio(
                "Select your answer:",
                options=[f"{key}) {value}" for key, value in options],
                key=f"answer_{question_data['id']}"
            )
            
            # Extract selected answer key
            user_answer_key = selected[0]
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("⏮️ Previous") and current_idx > 0:
                    st.session_state.current_quiz["current_question"] -= 1
                    st.rerun()
            with col2:
                if current_idx < len(questions) - 1:
                    if st.button("⏭️ Next"):
                        # Record answer
                        st.session_state.current_quiz["answers"][question_data['id']] = user_answer_key
                        st.session_state.current_quiz["current_question"] += 1
                        st.rerun()
                else:
                    if st.button("✅ Submit Quiz"):
                        # Record final answer
                        st.session_state.current_quiz["answers"][question_data['id']] = user_answer_key
                        
                        # Calculate score
                        correct = sum(
                            1 for q in questions
                            if st.session_state.current_quiz["answers"].get(q['id']) == 
                            next(key for key, val in [
                                ("A", q['correct_answer']),
                                ("B", q['wrong1']),
                                ("C", q['wrong2'])
                            ] if val == q['correct_answer'])
                        )
                        
                        score = int((correct / len(questions)) * 100)
                        complete_quiz_session(quiz["session_id"], score)
                        
                        # Record answers in database
                        for q in questions:
                            user_ans = st.session_state.current_quiz["answers"].get(q['id'], "")
                            is_correct = user_ans == next(key for key, val in [
                                ("A", q['correct_answer']),
                                ("B", q['wrong1']),
                                ("C", q['wrong2'])
                            ] if val == q['correct_answer'])
                            record_quiz_answer(q['id'], user_ans, is_correct, q['flashcard_id'])
                        
                        # Mark as completed
                        st.session_state.current_quiz["completed"] = True
                        st.session_state.current_quiz["current_question"] = 0
                        
                        st.success(f"Quiz completed! Your score: {score}%")
                        st.balloons()
                        st.rerun()
            with col3:
                if st.button("❌ Cancel Quiz"):
                    cursor.execute("DELETE FROM quiz_sessions WHERE id=?", (quiz["session_id"],))
                    cursor.execute("DELETE FROM quiz_questions WHERE session_id=?", (quiz["session_id"],))
                    conn.commit()
                    del st.session_state.current_quiz
                    st.rerun()

# --- MAIN APP ---
def main():
    st.set_page_config(
        page_title="Learn-with-me App",
        page_icon="📚",
        layout="centered"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Generate Flashcards", "Manage Flashcards", "Leitner Study", "Quiz Mode"],
        index=0
    )
    
    # Display the selected page
    if page == "Generate Flashcards":
        generate_flashcards_page()
    elif page == "Manage Flashcards":
        manage_flashcards_page()
    elif page == "Leitner Study":
        leitner_study_page()
    elif page == "Quiz Mode":
        quiz_page()

if __name__ == "__main__":
    main()