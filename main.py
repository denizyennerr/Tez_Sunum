import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to my master thesis web page! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    **👈 Select a page from the sidebar** to access the Presentation, Thesis or Demo!
    ### Want to learn more?
    - Check out code on [GitHub](https://github.com/denizyennerr/Tez)
    - Ask any question via [mail](mailto:[EMAIL_ADDRESS])
"""
)