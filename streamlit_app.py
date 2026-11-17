import os
import tempfile
import uuid

import requests
import streamlit as st

from vision import analyze_property_image


# ==================================================
# CONFIGURATION
# ==================================================

API_URL = os.getenv("API_URL", "http://backend:8000").rstrip("/")

API_SECRET_KEY = os.getenv("API_SECRET_KEY")


def new_memory():
    """Empty search preferences (the backend keeps the real memory)."""

    return {
        "locations": [],
        "property_type": None,
        "bedrooms": None,
        "min_price": None,
        "max_price": None
    }


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Real Estate AI Assistant",
    page_icon="🏡",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

def reset_session():
    """Start a fresh conversation with a new backend session."""

    st.session_state.api_session_id = str(uuid.uuid4())
    st.session_state.chat_messages = []
    st.session_state.memory = new_memory()
    st.session_state.image_result = None


if "api_session_id" not in st.session_state:
    reset_session()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🏡 Real Estate AI")

    st.write("Your AI-powered Mumbai property assistant.")

    st.divider()

    st.subheader("Search preferences")

    # Placeholder so the preferences can be refreshed after each answer
    memory_box = st.empty()
    memory_box.json(st.session_state.memory)

    st.divider()

    if st.button("Clear conversation"):
        reset_session()
        st.rerun()


# ==================================================
# MAIN PAGE
# ==================================================

st.title("🏡 Real Estate AI Assistant")

st.caption(
    "Find properties, compare projects and explore amenities "
    "using your AI assistant."
)


# ==================================================
# PROPERTY IMAGE ANALYSIS
# ==================================================

with st.expander("📷 Property image analysis"):

    uploaded_image = st.file_uploader(
        "Upload a property photograph or floor plan",
        type=["jpg", "jpeg", "png", "webp"]
    )

    image_question = st.text_input(
        "What would you like to know about this image?",
        value="Describe this property image."
    )

    if uploaded_image is None:
        st.session_state.image_result = None

    else:

        # A fixed pixel width works on every Streamlit version
        # (use_column_width is deprecated, use_container_width is newer)
        st.image(
            uploaded_image,
            caption="Uploaded property image",
            width=500
        )

        if st.button("Analyse image"):

            suffix = os.path.splitext(uploaded_image.name)[1]
            image_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(uploaded_image.getvalue())
                    image_path = temp_file.name

                with st.spinner("Qwen is analysing the image..."):
                    st.session_state.image_result = analyze_property_image(
                        image_path,
                        image_question
                    )

            except Exception as error:
                st.session_state.image_result = None
                st.error(str(error))

            finally:
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)

    # Stored in session state so it survives chat reruns
    if st.session_state.image_result:
        st.markdown("### AI analysis")
        st.write(st.session_state.image_result)


# ==================================================
# CHAT
# ==================================================

# Display chat history
for message in st.session_state.chat_messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask about Mumbai properties...")

if question:

    # Display and store the user message
    st.session_state.chat_messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Send the request to FastAPI
    with st.chat_message("assistant"):

        with st.spinner("AI is processing your request..."):

            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "session_id": st.session_state.api_session_id,
                        "question": question
                    },
                     headers={
                         "X-API-Key": API_SECRET_KEY
                    },
                    timeout=300
                )
                response.raise_for_status()

                result = response.json()
                answer = result["answer"]

                st.markdown(answer)

                # Keep the sidebar in sync with the backend's memory
                st.session_state.memory = result.get(
                    "memory",
                    st.session_state.memory
                )
                memory_box.json(st.session_state.memory)

            except requests.RequestException as error:
                answer = f"Backend connection error: {error}"
                st.error(answer)

            except (KeyError, ValueError) as error:
                answer = f"Unexpected API response: {error}"
                st.error(answer)

    # Save the assistant response
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": answer
    })