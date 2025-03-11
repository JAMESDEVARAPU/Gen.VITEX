import os
import yt_dlp
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss
import json
from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key="YOURS API KEY"
)

def download_audio(url, output_path="."):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'verbose': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            base, _ = os.path.splitext(filename)
            new_file = f"{base}.mp3"
            print(f"Audio downloaded successfully: {new_file}")
            return new_file
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
def get_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
def get_chunks(text, chunk_size=3):
    sentences = text.split(". ")
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def vectorize_chunks(chunks, model):
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype("float32")

def save_index(index, filename):
    faiss.write_index(index, filename)

def load_index(filename):
    return faiss.read_index(filename)

def get_similar(query, model, index, chunks, K=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, K)
    results = [(chunks[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    return results

def generate_summary_and_key_points(text, summarizer):
    summary = summarizer(text, max_length=150, min_length=100, do_sample=False)[0]['summary_text']
    key_points = summary.split(". ")[:5]
    return summary, key_points

def generate_related_questions(summary, num_questions=5):
    prompt = f"""
    Given the following summary, generate {num_questions} related questions:

    Summary: {summary}

    Generate {num_questions} questions that are directly related to the content of the summary. The questions should encourage further exploration of the topic.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    questions = chat_completion.choices[0].message.content.strip().split('\n')
    return questions


    return chat_completion.choices[0].message.content

# Main execution
if __name__ == "__main__":
    # Step 1: Download audio from YouTube
    video_url = input("Enter the YouTube video URL: ")
    audio_file_path = download_audio(video_url)

    if audio_file_path:
        # Step 2: Load the Whisper model for transcription
        whisper_model = whisper.load_model("base")

        # Transcribe the audio
        transcription = whisper_model.transcribe(audio_file_path, fp16=False, language="English")
        transcribed_text = transcription['text']
        print("Transcription:", transcribed_text)

        # Step 3: Load a pre-trained SentenceTransformer model
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode the transcription text into a vector
        embedding = sentence_model.encode(transcribed_text)
        embedding = np.array(embedding)

        print("Embedding vector:", embedding)
        print("Vector shape:", embedding.shape)

        # Initialize the model for summarization
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Process text into chunks
        chunks = get_chunks(transcribed_text)

        # Generate embeddings for chunks
        chunk_embeddings = vectorize_chunks(chunks, sentence_model)

        # Create FAISS index
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)
        print(f"Total chunks indexed: {index.ntotal}")

        # Save FAISS index
        index_file = "faiss_index.bin"
        save_index(index, index_file)

        # Generate summary and key points
        summary, key_points = generate_summary_and_key_points(transcribed_text, summarizer)

        print("\nSummary of the Text:")
        print(summary)
        print("\nKey Points:")
        for i, point in enumerate(key_points, 1):
            print(f"{i}. {point}")

        # Generate related questions
        related_questions = generate_related_questions(summary)
        print("\nRelated Questions:")
        for i, question in enumerate(related_questions, 1):
            print(f"{i}. {question}")

        # Perform a query
        query = "How does chunking help with text search?"
        results = get_similar(query, sentence_model, index, chunks, K=3)

        print("\nQuery:", query)
        print("\nTop matching chunks:")
        for chunk, dist in results:
            print(f"Chunk: {chunk} (Distance: {dist:.2f})")

        # Save chunks
        with open("chunks.pickle", "w") as w:
            w.write(json.dumps(chunks))

        
       
    else:
        print("Failed to download audio. Please check the URL and try again.")
