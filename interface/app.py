'''
No changes made so far
'''

import streamlit as st

st.markdown("# Fake Memories Detection or Classification")
st.markdown("### Authors: Rodolphe, Cynthia, Raghda")
st.markdown("***This application is deep learning related. It receives a user story and tells if he/she has a fake memory or not***")

story = st.text_area(
    label = "Please, enter your story here.",
    value = 'write your story here',
)

st.markdown("### Thank you very much for testing our application")
