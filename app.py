import io
import os
from datetime import timedelta
from functools import lru_cache, wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash

from src.helper import download_hugging_face_embeddings
from src.memory import ChatRepository
from src.prompt import system_prompt


app = Flask(__name__)

load_dotenv()
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(
    days=int(os.environ.get("CHAT_MEMORY_DAYS", "30"))
)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

INDEX_NAME = "medical-chatbot"
AUTH_USER_KEY = "auth_user_id"
RECENT_HISTORY_LIMIT = 20
MEMORY_DB_PATH = Path(
    os.environ.get(
        "CHAT_MEMORY_DB",
        Path(__file__).resolve().parent / "data" / "chat_memory.db",
    )
)
chat_repository = ChatRepository(MEMORY_DB_PATH)


def get_current_user():
    user_id = session.get(AUTH_USER_KEY)
    if not user_id:
        return None
    return chat_repository.get_user_by_id(user_id)


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not get_current_user():
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def get_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env.")
    return api_key


def get_pinecone_api_key() -> str:
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in .env.")
    return api_key


def form_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def convert_rows_to_history(rows):
    history = []
    for row in rows:
        content = str(row["content"]).strip()
        if not content:
            continue
        if row["role"] == "human":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))
    return history


def build_chat_page(user, conversation_id=None):
    selected_conversation = None
    if conversation_id:
        selected_conversation = chat_repository.get_conversation(user["id"], conversation_id)
        if not selected_conversation:
            flash("That conversation was not found in your account.")
            return redirect(url_for("index"))

    if not selected_conversation:
        selected_conversation = chat_repository.get_most_recent_conversation(user["id"])

    if not selected_conversation:
        new_conversation_id = chat_repository.create_conversation(user["id"])
        selected_conversation = chat_repository.get_conversation(user["id"], new_conversation_id)

    conversations = chat_repository.list_conversations(user["id"])
    messages = chat_repository.get_messages(user["id"], selected_conversation["id"])
    return render_template(
        "chat.html",
        user=user,
        conversations=conversations,
        current_conversation=selected_conversation,
        messages=messages,
    )


@lru_cache(maxsize=1)
def get_rag_chain():
    pinecone_api_key = get_pinecone_api_key()
    openai_api_key = get_openai_api_key()

    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    try:
        embeddings = download_hugging_face_embeddings()
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Hugging Face embedding model. "
            "Make sure the machine can reach huggingface.co or that the model is cached locally."
        ) from exc

    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to the Pinecone index '{INDEX_NAME}'. "
            "Check the API key and confirm the index already exists."
        ) from exc

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chat_model = ChatOpenAI(model="gpt-4o")
    contextualize_question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, rewrite the question "
                "so it can be understood on its own. Do not answer it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        chat_model,
        retriever,
        contextualize_question_prompt,
    )
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, question_answer_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_current_user():
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = chat_repository.get_user_by_username(username)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.")
        else:
            session.clear()
            session.permanent = True
            session[AUTH_USER_KEY] = user["id"]
            return redirect(url_for("index"))

    return render_template("auth.html")


@app.route("/register", methods=["POST"])
def register():
    if get_current_user():
        return redirect(url_for("index"))

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if len(username) < 3:
        flash("Username must be at least 3 characters long.")
        return redirect(url_for("login"))
    if len(password) < 6:
        flash("Password must be at least 6 characters long.")
        return redirect(url_for("login"))
    if password != confirm_password:
        flash("Passwords do not match.")
        return redirect(url_for("login"))

    created = chat_repository.create_user(username, generate_password_hash(password))
    if not created:
        flash("That username is already taken.")
        return redirect(url_for("login"))

    flash("Account created. Please log in.")
    return redirect(url_for("login"))


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return build_chat_page(get_current_user())


@app.route("/conversation/<conversation_id>")
@login_required
def open_conversation(conversation_id):
    return build_chat_page(get_current_user(), conversation_id=conversation_id)


@app.route("/conversation/new", methods=["POST"])
@login_required
def new_conversation():
    user = get_current_user()
    conversation_id = chat_repository.create_conversation(user["id"])
    return redirect(url_for("open_conversation", conversation_id=conversation_id))


@app.route("/get", methods=["POST"])
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    conversation_id = request.form.get("conversation_id", "").strip()

    if not msg:
        return jsonify({"error": "Please enter a message."}), 400
    if len(msg) > 4000:
        return jsonify({"error": "Message is too long. Please keep it under 4000 characters."}), 400

    user = get_current_user()
    conversation = chat_repository.get_conversation(user["id"], conversation_id) if conversation_id else None
    if not conversation:
        conversation_id = chat_repository.create_conversation(user["id"])
        conversation = chat_repository.get_conversation(user["id"], conversation_id)

    history = chat_repository.load_recent_history(
        user["id"],
        conversation["id"],
        limit=RECENT_HISTORY_LIMIT,
    )

    try:
        response = get_rag_chain().invoke({"input": msg, "chat_history": history})
    except RuntimeError as exc:
        app.logger.exception("Chatbot configuration error")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Unexpected chatbot failure")
        return jsonify({"error": "The chatbot could not process your request right now."}), 500

    answer = str(response.get("answer", "")).strip()
    final_answer = answer or "I couldn't find an answer for that."
    chat_repository.add_exchange(user["id"], conversation["id"], msg, final_answer)
    updated_conversation = chat_repository.get_conversation(user["id"], conversation["id"])

    return jsonify(
        {
            "answer": final_answer,
            "conversation": chat_repository.serialize_conversation(updated_conversation),
        }
    )


@app.route("/conversation/<conversation_id>/rename", methods=["POST"])
@login_required
def rename_conversation(conversation_id):
    user = get_current_user()
    title = " ".join(request.form.get("title", "").strip().split())
    if not title:
        return jsonify({"error": "Conversation title cannot be empty."}), 400
    if len(title) > 80:
        return jsonify({"error": "Conversation title is too long (max 80 characters)."}), 400

    renamed = chat_repository.rename_conversation(user["id"], conversation_id, title)
    if not renamed:
        return jsonify({"error": "Conversation not found."}), 404

    conversation = chat_repository.get_conversation(user["id"], conversation_id)
    return jsonify({"conversation": chat_repository.serialize_conversation(conversation)})


@app.route("/conversation/<conversation_id>/delete", methods=["POST"])
@login_required
def delete_conversation(conversation_id):
    user = get_current_user()
    deleted = chat_repository.delete_conversation(user["id"], conversation_id)
    if not deleted:
        return jsonify({"error": "Conversation not found."}), 404

    return jsonify({"redirect_url": url_for("index")})


@app.route("/conversation/<conversation_id>/pin", methods=["POST"])
@login_required
def pin_conversation(conversation_id):
    user = get_current_user()
    pinned = form_bool(request.form.get("pinned"), default=True)
    updated = chat_repository.set_conversation_pin(user["id"], conversation_id, pinned)
    if not updated:
        return jsonify({"error": "Conversation not found."}), 404

    conversation = chat_repository.get_conversation(user["id"], conversation_id)
    return jsonify({"conversation": chat_repository.serialize_conversation(conversation)})


@app.route("/conversation/<conversation_id>/clear", methods=["POST"])
@login_required
def clear_conversation(conversation_id):
    user = get_current_user()
    cleared = chat_repository.clear_conversation_messages(user["id"], conversation_id)
    if not cleared:
        return jsonify({"error": "Conversation not found."}), 404

    conversation = chat_repository.get_conversation(user["id"], conversation_id)
    return jsonify(
        {
            "conversation": chat_repository.serialize_conversation(conversation),
            "messages": [],
        }
    )


@app.route("/conversation/<conversation_id>/regenerate", methods=["POST"])
@login_required
def regenerate_conversation_reply(conversation_id):
    user = get_current_user()
    conversation = chat_repository.get_conversation(user["id"], conversation_id)
    if not conversation:
        return jsonify({"error": "Conversation not found."}), 404

    messages = chat_repository.get_messages(user["id"], conversation_id)
    if not messages:
        return jsonify({"error": "No messages found in this conversation."}), 400
    if messages[-1]["role"] != "ai":
        return jsonify({"error": "Regenerate is available after a bot response."}), 400

    last_human_index = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "human":
            last_human_index = idx
            break

    if last_human_index is None:
        return jsonify({"error": "No user message found to regenerate."}), 400

    user_question = str(messages[last_human_index]["content"]).strip()
    if not user_question:
        return jsonify({"error": "Last user message is empty."}), 400

    chat_history = convert_rows_to_history(messages[:last_human_index])
    try:
        response = get_rag_chain().invoke({"input": user_question, "chat_history": chat_history})
    except RuntimeError as exc:
        app.logger.exception("Chatbot configuration error during regenerate")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Unexpected chatbot failure during regenerate")
        return jsonify({"error": "The chatbot could not regenerate a response right now."}), 500

    answer = str(response.get("answer", "")).strip()
    final_answer = answer or "I couldn't find an answer for that."
    replaced = chat_repository.replace_last_ai_message(user["id"], conversation_id, final_answer)
    if not replaced:
        return jsonify({"error": "Could not update the last bot response."}), 500

    refreshed_conversation = chat_repository.get_conversation(user["id"], conversation_id)
    return jsonify(
        {
            "answer": final_answer,
            "conversation": chat_repository.serialize_conversation(refreshed_conversation),
        }
    )


@app.route("/transcribe", methods=["POST"])
@login_required
def transcribe_voice():
    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "Please record some audio first."}), 400

    try:
        client = OpenAI(api_key=get_openai_api_key())
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "Recorded audio is empty. Please try again."}), 400

    filename = audio_file.filename or "voice-query.webm"
    language = request.form.get("language", "").strip()

    models = ["gpt-4o-mini-transcribe", "whisper-1"]
    transcript = ""
    last_error = None
    for model in models:
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = filename
            request_args = {
                "model": model,
                "file": audio_buffer,
            }
            if language:
                request_args["language"] = language

            transcription = client.audio.transcriptions.create(**request_args)
            transcript = getattr(transcription, "text", "").strip()
            if transcript:
                break
        except Exception as exc:
            last_error = exc
            continue

    if not transcript:
        if last_error:
            app.logger.error("Voice transcription failed after trying fallback models: %s", last_error)
            return jsonify({"error": "Voice transcription failed. Please try again."}), 500
        return jsonify({"error": "No speech was detected in that recording."}), 422

    return jsonify({"text": transcript})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
