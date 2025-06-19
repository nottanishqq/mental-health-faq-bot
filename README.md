# Mental Health FAQ Chatbot
A smart, NLP-powered chatbot designed to answer frequently asked questions about mental health. This application uses a Word2Vec model to understand the semantic meaning of user questions and provide the most relevant answers from a given dataset. The user interface is built with Streamlit for a clean, interactive experience.

![image](https://github.com/user-attachments/assets/f8df2873-fc33-41c6-87c5-a075fb25a944)

# ✨ Features
- Natural Language Understanding: Powered by a Word2Vec model to find answers based on meaning, not just keywords.
- Interactive UI: A user-friendly, real-time chat interface built with Streamlit.
- Easy to Train: A separate training script allows you to train the bot on your own custom FAQ dataset (.csv).
- Robust & Scalable: The architecture separates the training process from the application logic, making it efficient and ready for deployment.
- Personalized Experience: Features custom sign-off messages and a clean, minimalist design.

# 🏗️ Project Architecture
This project is divided into three core components for a clean separation of concerns:

1. train.py (The Trainer): This is a command-line script responsible for a one-time task:
  - Reading a raw CSV dataset of questions and answers.
  - Preprocessing the text data (lemmatization, stop-word removal, etc.).
  - Training a Word2Vec model on the questions.
  - Generating vector representations for all questions.
  - Saving the trained model, question vectors, and processed Q&A data into the model_assets/ directory.

2. faq_bot.py (The Inference Engine): This file contains the FaqBot class, which acts as the "brain" of the application.
  - It loads the pre-trained artifacts from the model_assets/ directory.
  - It provides a single method, get_answer(), which takes a user's question, processes it, and compares it against the knowledge base to find the most similar question and return its answer.
  - It has no knowledge of the user interface; its sole responsibility is answering questions.

3. streamlit_app.py (The User Interface): This script is the user-facing application.
  - It loads the FaqBot class once at startup.
  - It creates the chat interface using Streamlit components.
  - It handles user input, checks for UI-specific commands (like quit), and calls the bot's get_answer() method.
  - It displays the bot's responses and the final goodbye message.

# 🚀 Setup and Installation
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
  - Python 3.8 or newer
  - pip (Python package installer)
2. Clone the Repository

Note: Replace your-username/your-repository-name in the command below with your actual GitHub path. You can find this URL by clicking the green "Code" button       on your repository's main page.

![image](https://github.com/user-attachments/assets/3876f4d6-97ff-4684-93f6-a2d07993e971)

3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.
   
![image](https://github.com/user-attachments/assets/9ae0aa87-dfc0-480d-b065-33b14a25fcbe)

4.  Install Dependencies

Install all the required Python packages from the requirements.txt file.
    
![image](https://github.com/user-attachments/assets/a07bbb67-c70d-4661-bc82-b390e34d326c)

# 🛠️ Usage
Running the chatbot is a two-step process:

### Step 1: Train the Model

First, you need to train the bot on your dataset. Run the train.py script from your terminal, providing the path to your CSV file. This command only needs to be run once.

Your CSV file must contain Question and Answer columns.

![image](https://github.com/user-attachments/assets/dc642e02-1ccb-4e06-9aa4-2835c74cdf1e)

This will create a model_assets/ directory containing the trained model and processed data.

### Step 2: Run the Streamlit Application

Once the training is complete and the model_assets/ directory exists, you can start the user interface.

![image](https://github.com/user-attachments/assets/49edab29-df80-4ff9-a87c-034ad5cc5066)

Your web browser will automatically open to the chat application, ready for you to use.

# 📁 File Structure

![image](https://github.com/user-attachments/assets/aa0d1112-fde1-48cc-8e91-88468536c0af)

# 🔧 Technologies Used
### Backend & NLP:
- Python
- NLTK: For natural language processing tasks.
- Gensim: For training and using the Word2Vec model.
- Pandas: For data manipulation.
- NumPy: For numerical operations.
- Scikit-learn: For calculating cosine similarity.
### Frontend:
- Streamlit: For creating the interactive web application UI.


