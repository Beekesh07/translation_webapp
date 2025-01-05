import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the Hugging Face model name
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load the model and tokenizer directly from Hugging Face
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit App
st.title("Translation App")
st.write("Translate English text to Hindi using a pre-trained model from Hugging Face.")

# Input text
input_text = st.text_area("Enter text in English:", "")

if st.button("Translate"):
    if input_text.strip():
        # Tokenize the input text
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

        # Perform translation
        outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the translated text
        st.write("### Translated Text:")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
