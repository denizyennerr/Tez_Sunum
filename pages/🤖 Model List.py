import os
import streamlit as st


def get_model_paths_grouped(base_path=None) -> dict:
    if base_path is None:
        base_path = os.path.join(os.getcwd(), "model_paths")

    grouped = {}

    if not os.path.exists(base_path):
        return grouped

    for folder in os.listdir(base_path):
        models_path = os.path.join(base_path, folder, "models")

        if os.path.isdir(models_path):
            grouped[folder] = [
                os.path.join(models_path, file)
                for file in os.listdir(models_path)
            ]

    return grouped


# --- UI ---
st.title("🤖 List of Models")

models_dict = get_model_paths_grouped()

if not models_dict:
    st.info("No models found. Please check the 'model_paths' folder.")
else:
    for folder, models in models_dict.items():
        with st.expander(f"📁 {folder}", expanded=False):
            if len(models) == 0:
                st.write("Model yok")
            else:
                for model in models:
                    model_name = os.path.basename(model)
                    st.write(f"🤖 | {model_name}")