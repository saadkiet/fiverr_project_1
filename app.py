import streamlit
import pandas as pd
import torch
from transformers import pipeline
import sentencepiece
import streamlit as st

def app():
    st.title("Text Summarization  🤓")

    st.markdown("This is a Web application that Summarizes Text 😎")
    upload_file = st.file_uploader('Upload a file containing Text data')
    button = st.button("Summarize")
    st.cache(allow_output_mutation=True)
    def google_model():
        model_name = 'google/pegasus-large'
        summarizer = pipeline('summarization',model=model_name, tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1)
        return summarizer

    summarizer_model = google_model()


    def text_summary(text):
       summarized_text = summarizer_model(text, max_length=130, min_length=1,clean_up_tokenization_spaces=True,no_repeat_ngram_size=4)
       summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
       return summarized_text

    # Check to see if a file has been uploaded
    if upload_file is not None and button:
        st.success("Summarizing Text, Please wait...")
        # If it has then do the following:

        # Read the file to a dataframe using pandas
        df = pd.read_csv(upload_file)

        # Create a section for the dataframe header
        #st.header('Header of Dataframe')
        #st.write(df.head(10))
        df1 = df.copy()
        df1['summarized_text'] = df1['Dialog'].apply(text_summary)
        #if st.button('Show Data'):
         #   st.success("Summarizng data, Please wait...")
        df2 = df1[['Name','summarized_text']]
        st.write(df2.head(5))

        @st.cache
        def convert_df(dataframe):
            return dataframe.to_csv().encode('utf-8')

        csv = convert_df(df2)
        st.download_button(label="Download CSV", data=csv, file_name='summarized_output.csv', mime='text/csv')






if __name__ == "__main__":
    app()
