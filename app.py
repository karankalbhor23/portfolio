import streamlit as st
import plotly.graph_objects as go
import base64


# Main function for the app
def main():
    st.set_page_config(page_title="Karan Kalbhor - Portfolio", layout="wide")
    
    # Uncomment and provide path to add a background image
    # add_bg_from_local('path/to/your/image.png')

    # Custom CSS for a more stylish look
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
   }
    
    .big-font {
        font-size: 60px !important;
        font-weight: 700;
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .medium-font {
        font-size: 36px !important;
        font-weight: 300;
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .small-font {
        font-size: 16px !important;
        background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-font {
        font-size: 28px !important;
        font-weight: 700;
         background: linear-gradient(45deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #3a7bd5;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background-color: #3a7bd5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #00d2ff;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #f0f8ff;
        border: 1px solid #3a7bd5;
        border-radius: 5px;
    }
    
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg, #3a7bd5, #00d2ff);
    }
    
    .sidebar .widget-title {
        color: white !important;
    }
    
    .sidebar .sidebar-collapse-control {
        background-color: white !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Karan Kalbhor")
    selected = st.sidebar.radio("", ["Home", "Experience", "Projects", "Skills", "Contact"])

    # Content based on selection
    if selected == "Home":
        home_page()
    elif selected == "Projects":
        projects_page()
    elif selected == "Skills":
        skills_page()
    elif selected == "Experience":
        experience_page()
    elif selected == "Contact":
        contact_page()

def home_page():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<p class="big-font">Karan Kalbhor</p>', unsafe_allow_html=True)
        st.markdown('<p class="medium-font">AI/ML Engineer | Data Scientist</p>', unsafe_allow_html=True)
    
    
    
    st.write("---")
    
    st.markdown('<p class="section-font">About Me</p>', unsafe_allow_html=True)
    st.write("""
    Welcome to my portfolio! I'm a recent graduate with a strong foundation in Data Science, Machine Learning, and AI. 
    My passion lies in driving innovative solutions and staying at the forefront of AI advancements. 
    Explore my projects and experiences to see how I've applied my skills in real-world scenarios.
    """)
    
    # Adding some interactivity
    if st.button("📩 Contact Me"):
        st.success("Thanks for your interest! Head to the Contact page to reach out.")

def projects_page():
    st.markdown('<p class="section-font">Featured Projects</p>', unsafe_allow_html=True)
    
    projects = {
        "Chat Bot Crafter": """Developed a revolutionary platform, Chat Bot Crafter, that harnesses the power of Large Language Models 
(LLMs), Natural Language Processing (NLP), and Knowledge Graph Embeddings to transform chatbot 
development. This platform enables users to select from pre-trained LLMs, define chatbot behavior, and 
integrate domain-specific knowledge bases, streamlining chatbot creation for developers and businesses. 
The platform functions by first selecting a pre-trained LLM, which is then fine-tuned using descriptive 
prompts to define the chatbot's behavior. The fine-tuned model is then integrated with a knowledge graph, 
powered by FAISS and Hugging Face embeddings, to provide accurate and context-aware responses. Upon 
receiving user input, the platform retrieves relevant information from the knowledge graph and generates 
a response using the LLM, presenting it to the user through a Streamlit-powered interactive UI.
""",
        "Instagram Caption Generator": """Developed an advanced tool utilizing the Phi-3.5 Vision model by Microsoft to generate contextually relevant 
captions for images on social media platforms. Leveraging computer vision and natural language processing, 
I integrated multimodal model components to enable seamless interaction between image and text data. 
This project showcased my expertise in deep learning, multimodal model integration, and AI-driven content 
generation, highlighting my ability to work with cutting-edge technologies and deploy sophisticated machine 
learning models for practical applications. """,
        "Autonomous AI Agents for Game Development": """Developed an innovative project using the CrewAI framework, where I orchestrated a crew of AI agents to 
autonomously create basic Python games. The AI agents were designed with specific roles and goals, such 
as game design, coding, and debugging, to collaboratively build fully functional games. Each agent was 
equipped with memory and custom tools to enhance its capabilities, ensuring efficient task execution and 
seamless interaction. The project demonstrated the potential of multi-agent systems in automating complex 
development tasks and showcased the practical application of AI in software engineering. """,
        "Interactive PDF Analysis": """I built a Streamlit web application that empowers users to analyze PDFs interactively using cutting- edge NLP 
techniques. Users upload PDFs which are processed with Transformers to create a searchable knowledge base. 
The application's core functionality is a chat-like interface powered by OpenAI's GPT-3 Large Language Model 
(LLM). This allows users to ask questions and receive real-time, informative answers directly related to the 
uploaded PDF content. This project showcases my active development skills using Streamlit, implementation 
of advanced NLP techniques (RAG and Transformers), and utilization of powerful AI models (GPT-3) to create 
user-centric applications.
 """
    }
    
    for project, description in projects.items():
        with st.expander(project):
            st.write(description)
            st.button("Learn More", key=project)  # You can add functionality to these buttons

def skills_page():
    st.markdown('<p class="section-font">Technical Skills</p>', unsafe_allow_html=True)
    
    skill_categories = {
        "Programming Languages": ["Python", "SQL", "C++"],
        "Frameworks & Libraries": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Flask", "OpenCV", "LangChain"],
        "AI/ML Techniques": ["Machine Learning", "Deep Learning", "Computer Vision", "NLP", "Large Language Models", "Prompt Engineering", "RAG", "Fine-tuning"]
    }
    
    for category, skills in skill_categories.items():
        st.subheader(category)
        cols = st.columns(3)
        for i, skill in enumerate(skills):
            cols[i % 3].checkbox(skill, value=True)  # Creates an interactive checklist

    # Skill proficiency chart
    proficiencies = {
        "Python": 90,
        "Machine Learning": 85,
        "Deep Learning": 80,
        "Data Analysis": 85,
        "Computer Vision": 75
    }
    
    fig = go.Figure([go.Bar(
        x=list(proficiencies.values()),
        y=list(proficiencies.keys()),
        orientation='h',
        marker=dict(
            color='rgba(58, 123, 213, 0.6)',
            line=dict(color='rgba(58, 123, 213, 1.0)', width=3)
        )
    )])
    fig.update_layout(
        title="Skill Proficiency",
        xaxis_title="Proficiency (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def experience_page():
    st.markdown('<p class="section-font">Professional Experience</p>', unsafe_allow_html=True)
    
    experiences = [
        {
            "title": "AI/ML Engineer Intern",
            "company": "AspireIT",
            "period": "March 2024 - Present",
            "description": [
                """As an AI Engineer intern, I worked with various Large Language Models (LLMs) and successfully fine-tuned 
them using techniques such as LoRA and QLoRA. This involved analyzing the models' performance, identifying 
areas for improvement, and using techniques such as parameter-efficient fine-tuning to optimize their 
performance. As a result, I was able to increase the accuracy of sentiment analysis by 15% and improve the 
coherence and relevance of generated text by 20% .""",
               """During my internship, I gained experience working with various Large Language Models (LLMs), including 
GPT, Gemini, Llama, Mistral, and others. I used Hugging Face Transformers to fine-tune these models for 
specific tasks, such as question answering and summarization, and to deploy GEN-AI applications in 
production environments. I also gained experience working with other Hugging Face libraries, such as 
Datasets and Tokenizers, to preprocess and tokenize data for use with LLMs.""",
               """I also gained a basic understanding of Retrieval-Augmented Generation (RAG), a technique for improving the 
accuracy and relevance of GEN-AI applications by using external knowledge sources to augment the 
information generated by the LLM. I worked on a project that used RAG to improve the performance of a 
question answering system, and I gained a basic understanding of how to use external knowledge sources to 
enhance the accuracy and relevance of GEN-AI applications.""",
                "Gained experience in Retrieval-Augmented Generation (RAG) for enhancing GEN-AI applications"
            ]
        },
        {
            "title": "Data Science Intern",
            "company": "Pantech.AI",
            "period": "January 2024 – April 2024",
            "description": [
                "Machine Learning Model Development: Designed, developed, and optimized machine learning models for predictive analytics, utilizing algorithms such as linear regression, decision trees, and clustering techniques",
                "Data Analysis and Visualization: Conducted comprehensive data analysis using Python, R, and SQL. Utilized pandas, NumPy, and Matplotlib for data preprocessing, exploratory data analysis, and visualizing insights.",
                "Deep Learning Applications: Explored deep learning techniques, including neural networks and convolutional neural networks (CNNs), for tasks such as image classification and natural language processing. Applied frameworks like TensorFlow and PyTorch to develop and test deep learning models."
            ]
        }
    ]
    
    for exp in experiences:
        with st.expander(f"{exp['title']} at {exp['company']}"):
            st.write(f"**Period:** {exp['period']}")
            for point in exp['description']:
                st.write(f"- {point}")

def contact_page():
    st.markdown('<p class="section-font">Get In Touch</p>', unsafe_allow_html=True)
    
    st.write("I'm always open to new opportunities and collaborations. Feel free to reach out!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Email:** karankalbhor1390@gmail.com")
        st.markdown("**Phone:** +91 9322086245")
    with col2:
        st.markdown("**Location:** Pune, India")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/karankalbhor23/)")
    
    # Simple contact form
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit_button = st.form_submit_button("Send Message")
        
        if submit_button:
            st.success("Thank you for your message! I'll get back to you soon.")

if __name__ == "__main__":
    main()
