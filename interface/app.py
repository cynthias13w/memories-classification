import streamlit as st

st.markdown("# Fake Memories Detection or Classification")
st.markdown("### Authors: Rodolphe, Cynthia, Raghda")
st.markdown("***This application is deep learning related. It receives a user story and tells if he/she has a fake memory or not***")

story = st.text_area(
    label="Please, enter your story here.",
    value='write your story here',
)

if st.button("Check for Fake Memory"):
    # You would place your model prediction logic here.
    # Replace this placeholder with your actual model logic.
    fake_memory_prediction = predict_fake_memory(story)

    if fake_memory_prediction:
        st.success("The story may contain a fake memory.")
    else:
        st.success("The story appears to be genuine.")

st.markdown("### Thank you very much for testing our application")

def predict_fake_memory(story):
    # Replace this placeholder with your actual model prediction code.
    # You'll need to load your model and preprocess the text.
    # Then, return the prediction (True for fake, False for genuine).
    # Here's a simple placeholder that returns False:
    return False
