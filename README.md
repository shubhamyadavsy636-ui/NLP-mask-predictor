Mini BERT Masked Word Predictor

Live Demo
[Click Here](https://nlp-mask-predictor-g4bcvs5pedbsavxzmebjtx.streamlit.app/)

Project Overview

This project implements a Masked Language Model (MLM) using BERT to predict missing words in a sentence.
Given an input sentence with a [MASK] token, the model intelligently predicts the most probable words based on contextual understanding.
This project demonstrates end-to-end deployment of a transformer-based NLP model, from training to a publicly accessible web application.

Ex. The capital of india is [MASK]. 


Key Features
•	 Predicts masked words using BERT
•	 Context-aware understanding of sentences
•	 Displays Top-5 predictions
•	 Interactive web interface using Streamlit
•	 Fast inference using Hugging Face Transformers

Tech Stack
•	Python
•	PyTorch
•	Hugging Face Transformers
•	Datasets (Hugging Face)
•	Streamlit
•	Docker

 Model Details
•	Model: BERT (bert-base-uncased)
•	Task: Masked Language Modeling (MLM)
•	Training Dataset: WikiText-2
•	Fine-tuned using Hugging Face Trainer API


1. Install dependencies
pip install -r requirements.txt
2. Run the app
streamlit run app.py

 Deployment
•	Model hosted on Hugging Face Hub
•	Application deployed using Streamlit Cloud
•	Containerized using Docker

Learning Outcomes
•	Implemented transformer-based NLP model
•	Understood Masked Language Modeling (MLM)
•	Worked with Hugging Face ecosystem
•	Handled real-world deployment challenges
•	Built scalable and portable ML application

 Acknowledgements
•	Hugging Face Transformers
•	PyTorch
•	Streamlit





