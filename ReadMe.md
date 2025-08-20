# 📚 Learn-with-me: AI-Powered Flashcard APP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)

![SQLite](https://img.shields.io/badge/SQLite-Database-green)

![HuggingFace](https://img.shields.io/badge/HuggingFace-T5--base-orange)

A comprehensive AI-powered learning application that automatically generates flashcards from documents, implements spaced repetition with the Leitner algorithm, and provides interactive quizzes for effective learning.

## 🌟 Features

• 📄 **Document Processing** - Extract text from PDF, DOCX, and TXT files
• 🤖 **AI-Powered Flashcard Generation** - Uses Google's Flan-T5 model to create questions from text
• 📊 **Leitner Spaced Repetition System** - Intelligent scheduling for optimal learning
• 🎯 **Interactive Quizzes** - Test your knowledge with multiple-choice questions
• 💾 **SQLite Database** - Persistent storage of flashcards and progress
• 🎨 **Modern UI** - Clean, professional interface built with Streamlit

## 🚀 How to Use

1. 📤 **Upload a document** (PDF, DOCX, or TXT)
2. 🎨 **Generate flashcards** from the document content
3. 📚 **Review flashcards** using the Leitner spaced repetition system
4. ✏️ **Take quizzes** to test your knowledge
5. 📈 **Track your progress** with visual statistics

## 🛠️ Installation

1. 📥 **Clone the repository:**
git clone https://github.com/your-username/learn-with-me.git

cd learn-with-me

2. 🏗️ **Create a virtual environment:**
python -m venv venv

source venv/bin/activate # On Windows: venv\Scripts\activate

3. 📦 **Install dependencies:**
pip install -r requirements.txt

4. 🚀 **Run the application:**
streamlit run app.py

## 📁 Project Structure
Learn-With-Me
 
├── app.py # 🐍 Main application file

├── learning.db # 🗄️ SQLite database (created automatically)

├── requirements.txt # 📋 Python dependencies

└── README.md # 📖 This file


## 🎯 How It Works

### 1. 📄 Document Processing
The app can extract text from uploaded documents:
• 📑 **PDF files** using PyPDF2
• 📝 **Word documents** using docx2txt  
• 📄 **Text files** directly

### 2. 🤖 AI Question Generation
Uses Hugging Face's Flan-T5 model to automatically generate multiple-choice questions from extracted text. The model is prompted to create questions with one correct answer and two wrong options.

### 3. 📊 Leitner Spaced Repetition
Implements the proven Leitner system:
• 🗃️ Cards start in Box 0 (review daily)
• 📈 Move to higher boxes when answered correctly
• 📉 Move to lower boxes when answered incorrectly
• 📅 Review intervals increase with each box (1, 2, 7, 14, 30, 90 days)

### 4. 💾 Database Management
Uses SQLite to store:
• 📋 Flashcard content and metadata
• 📝 Quiz sessions and results
• 📊 Leitner box information and review schedules

## 💻 Code Explanation

### 🗄️ Database Setup
```
# Creates database tables for flashcards, quiz sessions, and questions
def initialize_database():
    cursor.execute("""CREATE TABLE IF NOT EXISTS flashcards 
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   question TEXT, correct_answer TEXT, ...)""")
```
### 📄 Text Extraction
```
# Extracts text from different file types
def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
```            
### 🤖 AI Question Generation
```  
# Uses Flan-T5 model to generate questions
def generate_qa(text: str, num_questions: int = 5):
    result = qa_pipeline(prompt, max_length=512, do_sample=True)
    question_data = parse_any_format(result[0]['generated_text'])
``` 
### 📊 Leitner Algorithm
``` 
# Moves cards between boxes based on performance
def update_leitner_box(card_id, is_correct):
    if is_correct:
        new_box = min(current_box + 1, 5)  # Move up
    else:
        new_box = max(current_box - 1, 0)  # Move down
```         
### 🎨 UI Components
The app features four main pages:

•🎨 Generate Flashcards - Upload documents and create flashcards

•📋 Manage Flashcards - Edit, delete, and organize existing cards 

•📚 Leitner Study - Study using spaced repetition system

•✏️ Quiz Mode - Take interactive quizzes

### ⚡ Performance Features
•⚡ Cached Model Loading - Hugging Face model is cached for faster reloads

•🗄️ Efficient Database Queries - Optimized SQL queries for quick data access

•📱 Responsive Design - Works on desktop and mobile devices

•📊 Progress Tracking - Visual statistics on learning progress

### 🔧 Technical Details
•🎨 Frontend - Streamlit for web interface

•🐍 Backend - Pure Python with SQLite database

•🤖 AI Model - Google's Flan-T5-base from Hugging Face

•📄 File Processing - PyPDF2, docx2txt for document extraction

•📈 Data Visualization - Plotly for charts and statistics

### 🚦 Development Status

✅ Complete Features:

•📄 Document upload and text extraction

•🤖 AI question generation

•📋 Flashcard management

•📊 Leitner spaced repetition

•✏️ Quiz system

•💾 Database persistence

🔧 Future Enhancements:
• 👤 User authentication system

•☁️ Cloud synchronization

•📤 Export/import functionality

•🧠 Additional AI models

•📱 Mobile app version

## 🤝 Contributing
### This is a student project, but contributions are welcome! 
Feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4.  🔄 Submit a pull request

## 👨‍💻 About the Developer

### Created by Tushar Jain as a semester break project to explore:
• 🤖 AI and natural language processing

• 🗄️ Database management with SQLite

• 🌐 Web application development with Streamlit

• 🎓 Educational technology and learning systems

## 🎓 Learning Outcomes
### Through this project, I learned:
• 🧠 How to integrate AI models into applications

• 🗄️ Database design and management with SQLite

• 🌐 Building interactive web apps with Streamlit

• 📊 Implementing spaced repetition algorithms

• 📄 File processing and text extraction techniques

• 🏗️ Software architecture and project organization

# ⭐ If you find this project helpful, please give it a star on GitHub!