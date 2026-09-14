
import streamlit as st
import pandas as pd
import joblib
import re



from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


import nltk 
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

st.caption('Made by Shaheer Rangrej')

st.set_page_config(
    page_title="CustomerBot",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="expanded"
)


STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


@st.cache_data
def load_dataset():

    return pd.read_csv(
        "./data/customer_support.csv"
    )


@st.cache_resource
def load_models():

    model = joblib.load(
        "./notebook/model.pkl"
    )

    tfidf = joblib.load(
        "./notebook/tfidf.pkl"
    )

    intent_encoder = joblib.load(
        "./notebook/intent_encoder.pkl"
    )

    embeddings = joblib.load(
        "./notebook/embeddings.pkl"
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return (
        model,
        tfidf,
        intent_encoder,
        embeddings,
        embedding_model
    )


df = load_dataset()

(
    model,
    tfidf,
    intent_encoder,
    embeddings,
    embedding_model
) = load_models()


def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    text = re.sub(
        r"(.)\1{1,}",
        r"\1",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def get_response(
    user_text,
    threshold=0.70
):

    cleaned_text = preprocess(
        user_text
    )

    if not cleaned_text:

        return {
            "response": "Please describe your issue so I can help you.",
            "intent": "unknown",
            "similarity": 0.0
        }

    user_tfidf = tfidf.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        user_tfidf
    )[0]

    predicted_intent = (
        intent_encoder
        .inverse_transform([prediction])[0]
    )

    filtered_indices = df[
        df["intent"] == predicted_intent
    ].index

    if len(filtered_indices) == 0:

        return {
            "response": (
                "I couldn't find a suitable "
                "solution for your issue. "
                "Please provide more details."
            ),
            "intent": str(predicted_intent),
            "similarity": 0.0
        }

    filtered_embeddings = embeddings[
        filtered_indices
    ]

    user_embedding = embedding_model.encode(
        [cleaned_text],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        user_embedding,
        filtered_embeddings
    )[0]

    best_position = similarities.argmax()

    best_score = similarities[
        best_position
    ]

    best_index = filtered_indices[
        best_position
    ]

    best_row = df.loc[
        best_index
    ]

    if best_score < threshold:

        return {
            "response": (
                "I'm not completely sure I understood "
                "your issue. Could you provide a little "
                "more information so I can help you better?"
            ),
            "intent": str(predicted_intent),
            "similarity": float(best_score)
        }

    return {
        "response": best_row["response"],
        "intent": str(predicted_intent),
        "similarity": float(best_score)
    }


def is_greeting(text):

    text = text.lower().strip()

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    }

    return text in greetings


def greeting_message():

    return (
        "Hello! 👋\n\n"
        "I'm **SupportAI**, your AI-powered customer "
        "support assistant.\n\n"
        "Tell me what you're experiencing and I'll "
        "try to find the most relevant solution for you."
    )


def is_thanks(text):

    text = text.lower().strip()

    thanks_phrases = {
        "thanks",
        "thank you",
        "thanks a lot",
        "thankyou",
        "thx",
        "ty",
        "appreciate it",
        "that's helpful",
        "that helps",
        "got it",
        "perfect",
        "great"
    }

    return text in thanks_phrases


def is_yes(text):

    text = text.lower().strip()

    yes_phrases = {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "of course",
        "yes please",
        "please",
        "i do",
        "i have another question"
    }

    return text in yes_phrases


def is_no(text):

    text = text.lower().strip()

    no_phrases = {
        "no",
        "no thanks",
        "no thank you",
        "that's all",
        "thats all",
        "that's it",
        "thats it",
        "nothing else",
        "nothing",
        "i'm good",
        "im good",
        "all good",
        "nope",
        "not now"
    }

    return text in no_phrases


def closing_message():

    return (
        "You're very welcome! 😊\n\n"
        "Thank you for your patience. "
        "I'm glad I could assist you.\n\n"
        "Have a great day! 👋"
    )


def anything_else_message():

    return (
        "I hope that helped! 😊\n\n"
        "Is there anything else I can help you with?"
    )


def thanks_message():

    return (
        "You're very welcome! 😊\n\n"
        "Is there anything else I can help you with?"
    )


def start_new_conversation():

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": greeting_message()
        }
    ]

    st.session_state.question_count = 0

    st.session_state.waiting_for_confirmation = False

    st.session_state.conversation_closed = False


if "messages" not in st.session_state:

    start_new_conversation()


if "question_count" not in st.session_state:

    st.session_state.question_count = 0


if "waiting_for_confirmation" not in st.session_state:

    st.session_state.waiting_for_confirmation = False


if "conversation_closed" not in st.session_state:

    st.session_state.conversation_closed = False


with st.sidebar:

    st.title("🤖 SupportAI")

    st.caption(
        "AI Customer Support Platform"
    )

    st.divider()

    st.subheader("Conversation")

    st.metric(
        "Questions",
        st.session_state.question_count
    )

    if st.button(
        "＋ New Conversation",
        use_container_width=True
    ):

        start_new_conversation()

        st.rerun()

    st.divider()

    st.subheader("AI System")

    st.write(
        "🧹 NLP preprocessing"
    )

    st.write(
        "📊 TF-IDF classification"
    )

    st.write(
        "🧠 Intent detection"
    )

    st.write(
        "🔎 Semantic similarity"
    )

    st.write(
        "💬 Response retrieval"
    )

    st.divider()

    st.caption(
        "Powered by Machine Learning"
    )

    st.divider()

    st.header("View My Latest Project")

    st.subheader(
        " 🤖 AI RAG Chatbot",
        help="A document-based AI chatbot that allows users to upload PDF and Markdown (.md) files and ask questions about their content."
    )

    st.link_button(
        " 🚀 View Project",
        "https://ai-chatbotrag.streamlit.app/",
    )

    st.divider()

    st.subheader(
        " 🎬 AI Movie Recommendation",
        help="Movie recommendation system using machine learning & streamlit python"
    )

    st.link_button(
        " 🚀 View Project",
        "https://moviegpt-ai.streamlit.app/",
    )

    st.divider()

    st.subheader(
        " 🩺 AI Diabetic Prediction",
        help="Predict whether a patient has a higher or lower risk of diabetes based on medical and demographic features."
    )

    st.link_button(
        " 🚀 View Project",
        "https://diabetic-ai.streamlit.app/",
    )

    st.divider()

    st.subheader(
        " 👨‍💼 AI Employee Attrition",
        help="Predict whether an employee is likely to Stay or Leave a company."
    )

    st.link_button(
        " 🚀 View Project",
        "https://ai-employee-attrtion.streamlit.app/"
    )

    st.divider()

    st.subheader(
        "💳 AI Fraud Detection",
        help="Machine Learning fraud detection system that analyzes financial transactions and predicts whether a transaction is Fraudulent or Legitimate."
    )

    st.link_button(
        " 🚀 View Project",
        "https://ai-employee-attrtion.streamlit.app/"
    )


st.title("Customer Support")

st.caption(
    "Describe your problem and let SupportAI find the most relevant answer."
)


if len(st.session_state.messages) == 1:

    st.info(
        "👋 Welcome! You can ask me about orders, "
        "payments, refunds, deliveries, accounts, "
        "or other customer-support issues."
    )

    st.subheader(
        "How can I help?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📦 Cancel purchase",
            use_container_width=True
        ):

            st.session_state.quick_message = (
                "I need to cancel purchase"
            )

            st.rerun()

    with col2:

        if st.button(
            "💳 Payment issue",
            use_container_width=True
        ):

            st.session_state.quick_message = (
                "My payment failed"
            )

            st.rerun()

    with col3:

        if st.button(
            "🚚 Address issue",
            use_container_width=True
        ):

            st.session_state.quick_message = (
                "there is an issue trying to correct the address"
            )

            st.rerun()


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "similarity" in message
        ):

            with st.expander(
                "View AI analysis"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Detected Intent",
                        message["intent"]
                    )

                with col2:

                    st.metric(
                        "Match Score",
                        f"{message['similarity']:.1%}"
                    )


user_input = st.chat_input(
    "Describe your issue..."
)


quick_message = st.session_state.pop(
    "quick_message",
    None
)


if quick_message:

    user_input = quick_message


if user_input:

    user_input = user_input.strip()

    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_input
            )

        st.session_state.question_count += 1

        with st.chat_message(
            "assistant"
        ):

            if is_greeting(
                user_input
            ):

                response = greeting_message()

                st.markdown(
                    response
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            elif is_no(user_input):

                if st.session_state.waiting_for_confirmation:

                    response = closing_message()

                    st.markdown(
                        response
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

                    st.session_state.waiting_for_confirmation = False

                    st.session_state.conversation_closed = True

                else:

                    response = (
                        "No problem! 😊 "
                        "If you need anything in the future, "
                        "I'm here to help."
                    )

                    st.markdown(
                        response
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

            elif is_yes(user_input):

                if st.session_state.waiting_for_confirmation:

                    response = (
                        "Of course! 😊 "
                        "Please tell me what you'd like help with."
                    )

                    st.markdown(
                        response
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

                    st.session_state.waiting_for_confirmation = False

                    st.session_state.conversation_closed = False

                else:

                    response = (
                        "Absolutely! 😊 "
                        "Please tell me what you'd like help with."
                    )

                    st.markdown(
                        response
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

            elif is_thanks(user_input):

                response = thanks_message()

                st.markdown(
                    response
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                st.session_state.waiting_for_confirmation = True

            else:

                with st.spinner(
                    "🔎 Finding the best solution..."
                ):

                    result = get_response(
                        user_input
                    )

                response = result["response"]

                st.markdown(
                    response
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "intent": result["intent"],
                        "similarity": result["similarity"]
                    }
                )

                follow_up = anything_else_message()

                st.markdown(
                    follow_up
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": follow_up
                    }
                )

                st.session_state.waiting_for_confirmation = True

                st.session_state.conversation_closed = False

