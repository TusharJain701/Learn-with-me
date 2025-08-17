# --- IMPORTS ---
import sqlite3
import re
from typing import Optional, List
import streamlit as st
import pandas as pd
import docx2txt
from PyPDF2 import PdfReader
import torch
from transformers import pipeline
import random

# --- SQLite SETUP ---

# Connect SQLite database file for storing flashcards
conn = sqlite3.connect("flashcards.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS flashcards 
               (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               question TEXT,
               correct_answer TEXT,
               wrong1 TEXT,
               wrong2 TEXT,
               difficulty TEXT
               )""")
conn.commit()

# --- TEXT PROCESSING FUNCTIONS ---
def extract_text(file):
    """
    Extracts text from uploaded files (PDF, DOCX, or TXT)
    """
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
    """
    Cleans and normalizes extracted text
    """
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

# --- QA GENERATION FUNCTIONS ---
@st.cache_resource
def load_qa_model():
    """
    Loads and caches the Hugging Face question generation model
    """
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # may use google/flan-t5-small due to lack of vram and cuda cores 
        device=device
    )
# Initialize the model pipeline
qa_pipeline = load_qa_model()  

def generate_qa(text: str, num_questions: int = 5) -> List[List[str]]:
    qa_pairs = []
    attempts = 0
    max_attempts = num_questions * 3
    
    # Split text into chunks for variation
    chunk_size = 1000
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] if len(text) > chunk_size else [text]
    
    while len(qa_pairs) < num_questions and attempts < max_attempts:
        attempts += 1
        
        try:
            # Select random chunk with context
            context_chunk = random.choice(text_chunks) if text_chunks else text
            
            # Create varied prompt templates
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
            
            # Only add valid and unique questions
            if question_data and not any(q[0] == question_data[0] for q in qa_pairs):
                qa_pairs.append(question_data)
                
        except Exception as e:
            st.warning(f"Error generating question: {str(e)}")
            continue
            
    if not qa_pairs and attempts >= max_attempts:
        st.error("Failed to generate flashcards after multiple attempts. The text might be too short or complex.")
    
    return qa_pairs[:num_questions]

def parse_any_format(qa_text: str) -> Optional[List[str]]:
    """
    More robust parser that handles multiple question format variations from LLM output
    """
    # Normalize the text first
    qa_text = qa_text.replace("\n", " ").strip()
    
    # Try to extract question and options using more flexible patterns
    question_match = re.search(r'^(.*?\?)', qa_text)
    if not question_match:
        return None
    
    question = question_match.group(1).strip()
    
    # Try to find options in various formats
    option_patterns = [
        r'[A-D][):.]?\s*(.*?)(?:\s*(?:[A-D][):.]|$))',  # A) option1 B) option2
        r'\d[.:]\s*(.*?)(?:\s*(?:\d[.:]|$))',          # 1. option1 2. option2
        r'-\s*(.*?)(?:\s*(?:-|$))',                    # - option1 - option2
        r'option\s*\w\s*:\s*(.*?)(?:\s*(?:option\s*\w\s*:|$))'  # option A: option1 option B: option2
    ]
    
    options = []
    for pattern in option_patterns:
        options = re.findall(pattern, qa_text, re.IGNORECASE)
        if len(options) >= 3:  # We need at least 3 options
            break
    
    if len(options) < 3:
        # Fallback - just take the text after question as options
        remaining_text = qa_text[len(question):].strip()
        if remaining_text:
            options = remaining_text.split()[:3]
        else:
            options = ["Correct answer", "Wrong option 1", "Wrong option 2"]
    
    # Try to identify correct answer (look for "answer:" or similar)
    answer_patterns = [
        r'answer\s*[:\-]\s*([A-D1-3])',
        r'correct\s*[:\-]\s*([A-D1-3])',
        r'right\s*[:\-]\s*([A-D1-3])'
    ]
    
    correct_idx = 0  # default to first option if we can't determine
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
    
    # Ensure we have at least 3 options
    while len(options) < 3:
        options.append(f"Option {len(options)+1}")
    
    correct = options[correct_idx]
    wrongs = [opt for i, opt in enumerate(options) if i != correct_idx]
    
    # Ensure we have at least 2 wrong options
    while len(wrongs) < 2:
        wrongs.append(f"Wrong option {len(wrongs)+1}")
    
    return [question, correct, wrongs[0], wrongs[1]]

# --- Helper functions for Flash Cards DB operations ---

def save_flashcard(question, correct, wrong1, wrong2, difficulty):
    cursor.execute(
        "INSERT INTO flashcards (question, correct_answer, wrong1, wrong2, difficulty) VALUES (?, ?, ?, ?, ?)",
        (question, correct, wrong1, wrong2, difficulty)
    )
    conn.commit()

def get_flashcards():
    df = pd.read_sql_query("SELECT * FROM flashcards", conn)
    return df

def update_flashcard(card_id, question, correct, wrong1, wrong2, difficulty):
    cursor.execute(
        "UPDATE flashcards SET question=?, correct_answer=?, wrong1=?, wrong2=?, difficulty=? WHERE id=?",
        (question, correct, wrong1, wrong2, difficulty, card_id)
    )
    conn.commit()

def delete_flashcard(card_id):
    cursor.execute("DELETE FROM flashcards WHERE id=?", (card_id,))
    conn.commit()

# --- PAGE CONFIG ---
st.set_page_config(page_title="📚",page_icon="📚")
# --- SIDE BAR NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Generate Flashcards", "View/Edit Flashcards"])

# --- PAGE: Generate Flashcards ---
if page == "Generate Flashcards":
    st.set_page_config(page_title="Generate Flashcards", layout="centered")
    st.title("📄 Flashcard Generator")
    st.info("Upload a document, choose how many questions you want, and then generate flashcards. Pick the ones you want to keep!")

    # File Upload
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
    
        # Numerical Input for Number of Questions
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
    
    # Display generated flashcards with option to keep/dispose and select difficulty
    if "qa_pairs" in st.session_state:
        st.subheader("📝 Generated Flashcards")
        flashcards_to_save = []  # Collect the ones the user wants to keep
        with st.form("save_flashcards_form"):
            for i, (question, correct, wrong1, wrong2) in enumerate(st.session_state.qa_pairs, 1):
                with st.expander(f"Flashcard {i}: {question}", expanded=False):
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Correct Answer:** {correct}")
                    st.markdown(f"**Option 1:** {wrong1}")
                    st.markdown(f"**Option 2:** {wrong2}")
                    difficulty = st.selectbox(f"Select difficulty for flashcard {i}", 
                                              ["Easy", "Medium", "Hard"], key=f"diff_{i}")
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
                    st.success("Selected flashcards saved!")
                   
                    # Fun animations!
                    st.balloons()
                else:
                    st.info("No flashcards selected to save.")

# --- PAGE: View/Edit Flashcards ---
elif page == "View/Edit Flashcards":
    st.set_page_config(page_title="View/Edit Flashcards", layout="centered")
    st.title("🗄️ Saved Flashcards")
    st.info("Edit or delete your saved flashcards below.")
    
    df = get_flashcards()
    if df.empty:
        st.warning("No flashcards saved yet.")
    else:
        # Display flashcards in individual expanders with edit options
        for index, row in df.iterrows():
            with st.expander(f"Flashcard {row['id']}: {row['question']}", expanded=False):
                # Create an editable form for each card
                with st.form(key=f"edit_form_{row['id']}"):
                    new_question = st.text_area("Question", row["question"], key=f"question_{row['id']}")
                    new_correct = st.text_input("Correct Answer", row["correct_answer"], key=f"correct_{row['id']}")
                    new_wrong1 = st.text_input("Wrong Option 1", row["wrong1"], key=f"wrong1_{row['id']}")
                    new_wrong2 = st.text_input("Wrong Option 2", row["wrong2"], key=f"wrong2_{row['id']}")
                    new_diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(row["difficulty"]), key=f"diff_{row['id']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        update_btn = st.form_submit_button("Update")
                    with col2:
                        delete_btn = st.form_submit_button("Delete")
                    
                    if update_btn:
                        update_flashcard(row['id'], new_question, new_correct, new_wrong1, new_wrong2, new_diff)
                        st.success("Flashcard updated!")
                        st.experimental_rerun()
                    if delete_btn:
                        delete_flashcard(row['id'])
                        st.success("Flashcard deleted!")
                        st.experimental_rerun()
