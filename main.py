import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from pytube import YouTube
from sklearn.metrics.pairwise import cosine_similarity
import openai
import assemblyai as aai

# Load environment variables
load_dotenv()
ASSEMBLY_AI_API_KEY = os.getenv('ASSEMBLY_AI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set API keys for AssemblyAI and OpenAI
aai.settings.api_key = ASSEMBLY_AI_API_KEY
openai.api_key = OPENAI_API_KEY

# PyTube function to download YouTube video audio
def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    
    # Check if file already exists
    if os.path.exists(file_name):
        return file_name
    
    os.rename(out_file, file_name)
    return file_name

# AssemblyAI speech-to-text function
def assemblyai_stt(audio_filename):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_filename)
    
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Error during transcription: {transcript.error}")
    
    return transcript.text

# Function to get embeddings using OpenAI's GPT-3
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try:
        response = openai.Embedding.create(input=[text], model=model)
        
        # Check if 'data' exists in response and is not empty
        if 'data' in response and response['data']:
            embedding = response['data'][0]['embedding']
            return embedding
        else:
            raise RuntimeError("Failed to create embeddings. Response data is empty or format unexpected.")
    
    except Exception as e:
        raise RuntimeError(f"OpenAI API error occurred: {str(e)}")

# Function to save embeddings to a file
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Function to load embeddings from a file
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to get the answer from OpenAI
def get_answer(query, context):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit application
st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")
st.title("Chat with Your Audio using LLM")

input_source = st.text_input("Enter the YouTube video URL")

embeddings_file = "embeddings.pkl"
transcription_file = "transcription.txt"

if input_source:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Your uploaded video")
        st.video(input_source)
        try:
            audio_filename = save_audio(input_source)
            if not os.path.exists(transcription_file):
                transcription = assemblyai_stt(audio_filename)
                st.info(transcription)

                # Save transcription
                with open(transcription_file, 'w') as f:
                    f.write(transcription)

                # Get embeddings using OpenAI's GPT-3
                embeddings = get_embedding(transcription)
                save_embeddings(embeddings, embeddings_file)
                st.success("Embeddings created and saved successfully!")
            else:
                st.success("Audio already processed and embeddings exist.")
        except Exception as e:
            st.error(f"Error processing the video: {e}")

    with col2:
        st.info("Chat Below")
        query = st.text_area("Ask your Query here...")
        if query and os.path.exists(embeddings_file):
            if st.button("Ask"):
                try:
                    st.info(f"Your Query is: {query}")
                    
                    # Load embeddings from the file
                    embeddings = load_embeddings(embeddings_file)
                    
                    # Get embedding for the query
                    query_embedding = get_embedding(query)
                    
                    # Compute cosine similarity
                    similarity = cosine_similarity([query_embedding], [embeddings])[0][0]
                    
                    # Threshold for similarity (tweak as needed)
                    threshold = 0.5
                    
                    if similarity > threshold:
                        st.success("The answer to your question is in the video transcript.")
                        
                        # Load the transcription from the file
                        with open(transcription_file, 'r') as f:
                            transcription = f.read()

                        # Get the answer from OpenAI
                        answer = get_answer(query, transcription)
                        st.info(f"Answer: {answer}")
                    else:
                        st.info("I don't know. The answer is not in the video transcript.")

                    st.write(f"Similarity score: {similarity}")
                except Exception as e:
                    st.error(f"Error querying the embeddings: {e}")
        elif query:
            st.warning("Please process the video first to create embeddings.")
