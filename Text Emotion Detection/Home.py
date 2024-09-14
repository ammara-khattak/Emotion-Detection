import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import joblib

from streamlit_option_menu import option_menu

import time

# Create a placeholder for the splash screen


def display_navigation():
    logo_url = "logo1.png"  # Replace with your logo URL or file path

# Create two columns: one for the logo and one for the navbar
    col1, col2 = st.columns([1, 10])

    with col1:
        st.image(logo_url, width=80)  # Adjust the logo size as needed

    with col2:
        selected = option_menu(
        menu_title= None,
        options= ["Home","About","Contact","FAQs"],
        icons=["house-door-fill", "info-circle-fill" , "envelope-at-fill", "clipboard-fill"],
        menu_icon="cast",
        default_index=0,
        orientation = "horizontal",
        styles= {
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#80ced6", "font-size": "25px"}, 
        "nav-link": {"font-size": "22px", "text-align": "centre", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#1a1aff"},
        }
        
    )
    return selected


pipe_lr = joblib.load(open("model/emotion_detec.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def home():

    placeholder = st.empty()

# Display the splash screen
    with placeholder.container():
        st.image("logo1.png", width=300)
        st.markdown(
        """
        <style>
        .splash-screen {
            position: fixed;
            width: 100%;
            height: 100%;
            background-color: white;
            color: black;
            display: flex;
            justify-content: initial;
            align-items: start;
            font-size: 55px;
            z-index: 9999;
        }
         .splash-screen img {
            width: 90px;  # Adjust the image size as needed
            margin-bottom: 20px;
        }
        </style>
        <div class="splash-screen">
            <div>
                <h1>SENTIMENT SENSE</h1>
                <p>Loading...</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Simulate a delay (e.g., loading time)
    time.sleep(1)  # Wait for 3 seconds

# Remove the splash screen and display the main content
    placeholder.empty()


    st.title("Sentiment Sense")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)
def about():
    st.title("About APPLICATION")
    st.write("Welcome to SentimentSense, your go-to platform for cutting-edge sentiment analysis! At SentimentSense, we harness the power of advanced artificial intelligence and machine learning to understand and interpret human emotions expressed in text. Whether it's social media posts, product reviews, or customer feedback, our tools provide deep insights to help you make data-driven decisions.")
    st.title("Mission")
    st.write("Our mission is to empower businesses, researchers, and individuals with the ability to gauge public sentiment accurately and efficiently. We strive to deliver robust and reliable sentiment analysis solutions that offer actionable insights, helping you stay ahead in today's fast-paced digital world.")
    st.title("What We Offer")
    st.write("Real-Time Sentiment Analysis: Instantly analyze and categorize sentiments from various text sources. Comprehensive Reports: Get detailed sentiment breakdowns with intuitive visualizations. Customizable Solutions: Tailor our sentiment analysis tools to fit your specific needs and industry. User-Friendly Interface: Enjoy a seamless experience with our easy-to-navigate platform.")
    st.title("Why Choose SentimentSense?")
    st.write("Accuracy: Our sophisticated algorithms deliver high precision in sentiment analysis. Speed: Get real-time results to stay on top of trends and feedback. Customization: Adapt our tools to suit your unique requirements. Expertise: Our team consists of seasoned professionals in AI, NLP, and data science, dedicated to providing the best service.")


# Example data for the cards
    st.title("ABOUT US:")
    cards_data = [
    {"title": "FATIMA JAWAID", "content": "As a frontend developer, my role involves designing and developing an interactive user interface using Streamlit for the sentiment analysis web app. I integrate the machine learning model, which is saved using Pickle, to provide real-time sentiment predictions based on user input. My focus is on creating a smooth user experience by handling inputs, displaying results, and ensuring efficient performance.", "image": "https://media.licdn.com/dms/image/D4D03AQGrfvbzbTK17A/profile-displayphoto-shrink_800_800/0/1705085431392?e=1728518400&v=beta&t=4xFwIRHzWY6y-34myMIrZ5nws1iYqfT9GzUMJd5iJdw"},
    {"title": "SANA MATEEN", "content": "As a Machine Learning Trainer with a focus on sentiment analysis, I specialize in developing and optimizing models that accurately interpret emotions from text. My role includes data preprocessing, selecting the right algorithms, and continuously improving model performance. I work closely with data scientists and engineers to integrate these models into applications, providing valuable insights and driving innovation.", "image": "https://images.meesho.com/images/products/272111405/rkhor_512.wep"},
    {"title": "AMMARA KHATTAK", "content": "In the sentiment analysis project, I handle documentation by creating user guides, detailing model integration with Pickle, explaining app features, and maintaining clear code comments for easy maintenance and updates. Maintaining well-structured and annotated code documentation to facilitate understanding and future maintenance, including comments and explanations for complex sections of the code.", "image": "https://media.licdn.com/dms/image/v2/D4D35AQGIRezb2LoqSg/profile-framedphoto-shrink_800_800/profile-framedphoto-shrink_800_800/0/1723820408445?e=1726873200&v=beta&t=kJpuO9IY5XgIP3qaroLOyVh95UxNb1bNqyOHShKSW4U"}
    ]

# CSS for card styling
    st.markdown(
    """
    <style>
        body {
        
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-color: #f4f4f4; /* Optional background color */
    }
     .card-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;  /* Space between cards */
        margin: 0 auto; /* Centers container horizontally */
        max-width: 100%; /* Prevents container from exceeding viewport width */
    }
    .card {
        
        border: 1px solid blue;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        width: 100%; /* Full width by default */
        max-width: 400px; /* Maximum width of card */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
    }
    .card img {
        
        justify-content: center;
        
        border-radius: 60px;
        border: 3px solid white;
        align-items: center;
        height: 125px;
        width: 125px;
    }
    .card-title {
        justify-content: center;
        font-size: 1.5em;
        font-weight: 600;
        margin-bottom: 10px;
        color: blue;
        
    }
    .card-content {
        font-size: 1em;
        text-align: justify;
    }
        @media (max-width: 480px) {
        .card-title {
            font-size: 1em; /* Further adjust title size */
        }
        .card-content {
            font-size: 0.8em; /* Further adjust content size */
        }
        .card img {
            height: 80px; /* Adjust image size */
            width: 80px;  /* Adjust image size */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('<div class="card-container">', unsafe_allow_html=True)

# Create cards using HTML and CSS
    for card in cards_data:
        st.markdown(
        f"""
        <div class="card">
            <img src="{card['image']}" alt="Image">
            <div class="card-title">{card['title']}</div>
            <div class="card-content">{card['content']}</div>
        </div>
        """,
        unsafe_allow_html=True
        )


def contact():
    st.header(":mailbox: Get In Touch With Us..!")
    contact_form = """
    <form action="https://formsubmit.co/fatimalodhi1425@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    local_css("style/sty.css")

def faq():
    st.title("Frequently Asked Question")

    # Questions and answers
    faq = [
    {
        "question": "1. What is SentimentSense?",
        "answer": "SentimentSense is a platform that uses AI and NLP to analyze and interpret emotions in text, providing insights from sources like social media, reviews, and feedback."
    },
    {
        "question": "2. How does sentiment analysis work?",
        "answer": "We use machine learning algorithms to evaluate text, identifying positive, negative, and neutral sentiments based on context, tone, and language patterns."
    },
    {
        "question": "3. What types of text can be analyzed?",
        "answer": "We can analyze social media posts, product reviews, customer feedback, survey responses, blogs, and news articles."
    },
    {
        "question": "4. Can I use this for free?",
        "answer": "Yes, our web application is for everyone ."
    },
    {
        "question": "5. Can SentimentSense analyze multiple languages?",
        "answer": "Yes, our platform supports multiple languages, making it versatile for global users."
    },
    {
        "question": "6. Is SentimentSense suitable for businesses?",
        "answer": "Yes, it’s ideal for businesses seeking to understand customer sentiment, improve satisfaction, and make data-driven decisions"
    },
    {
        "question": "7. Can I customize the analysis for my needs?",
        "answer": "Yes, our tools can be tailored to focus on aspects most relevant to your industry or objectives."
    },
    {
        "question": "8. How do I contact support?",
        "answer": ""
    },
    ]

    with open("sty1.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Displaying the FAQ
    for item in faq:
     with st.expander(item["question"]):
        st.write(item["answer"])

    

pages = {
    "Home": home,
    "About": about,
    "Contact": contact,
    "FAQs": faq
}



    
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)






def main():
    selected = display_navigation()
    page = pages[selected]
    page()


if __name__ == '__main__':
    main()
