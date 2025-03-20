import streamlit as st
import os
import warnings
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import time

# Warning control
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CrewAI Article Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004D40;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-text {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>✨ AI Article Generator ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>This app uses CrewAI to generate an article on your topic of choice. Multiple AI agents (Planner, Writer, and Editor) work together to create a well-structured article.</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    api_option = st.selectbox(
        "Select API Provider",
        ["OpenAI", "Groq", "Llama (via Ollama)"]
    )
    
    if api_option == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
        )
    elif api_option == "Groq":
        groq_api_key = st.text_input("Groq API Key", type="password")
        model_name = st.selectbox(
            "Select Groq Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
        )
    else:  # Llama
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        model_name = st.selectbox(
            "Select Llama Model",
            ["llama3", "llama3:8b", "llama3:70b"]
        )
    
    st.markdown("<h2 class='sub-header'>👥 The AI Crew</h2>", unsafe_allow_html=True)
    st.markdown("""
    - **Planner**: Researches the topic and creates an outline
    - **Writer**: Writes the article based on the plan
    - **Editor**: Polishes the article for publication
    """)

# Main content
st.markdown("<h2 class='sub-header'>🔍 Select Your Topic</h2>", unsafe_allow_html=True)

topic = st.text_input("What would you like your article to be about?", placeholder="e.g., Artificial Intelligence in Healthcare")

# Configure button based on form fill status
is_ready = False
button_message = ""

if not topic:
    button_message = "Please enter a topic"
elif api_option == "OpenAI" and not openai_api_key:
    button_message = "Please enter your OpenAI API key"
elif api_option == "Groq" and not groq_api_key:
    button_message = "Please enter your Groq API key"
else:
    is_ready = True
    button_message = "Generate Article"

# Display button with appropriate state
if st.button(button_message, disabled=not is_ready):
    if is_ready:
        try:
            # Configure LLM based on selection
            if api_option == "OpenAI":
                os.environ["OPENAI_API_KEY"] = openai_api_key
                llm = ChatOpenAI(model_name=model_name)
            elif api_option == "Groq":
                os.environ["OPENAI_API_KEY"] = groq_api_key
                os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
                llm = ChatOpenAI(model_name=model_name)
            else:  # Llama via Ollama
                from langchain_community.llms import Ollama
                llm = Ollama(model=model_name, base_url=ollama_url)
            
            # Create progress indicators
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            
            status_placeholder.markdown("<p class='info-text'>Starting the AI crew...</p>", unsafe_allow_html=True)
            progress = progress_placeholder.progress(0)
            
            # Define agents
            status_placeholder.markdown("<p class='info-text'>🧠 Creating the planning agent...</p>", unsafe_allow_html=True)
            progress.progress(10)
            
            planner = Agent(
                role="Planejador de conteúdo",
                goal=f"Planeje conteúdo envolvente e efetivamente preciso sobre {topic}",
                backstory=f"Você está planejando um artigo para o blog "
                        f"sobre o tópico: {topic}."
                        f"Você coleta informações que ajudam o "
                        f"público aprender alguma coisa "
                        f"e tomar decisões informadas. "
                        f"Seu trabalho é a base para "
                        f"o redator de conteúdo para escrever um artigo sobre este tópico.",
                allow_delegation=False,
                verbose=True,
                llm=llm
            )
            
            status_placeholder.markdown("<p class='info-text'>✍️ Creating the writing agent...</p>", unsafe_allow_html=True)
            progress.progress(20)
            
            writer = Agent(
                role="Escritor de conteúdo",
                goal=f"Escreva um perspicaz e efetivamente preciso "
                    f"artigo de opinião sobre o tema: {topic}",
                backstory=f"Você está trabalhando em um texto "
                        f"um novo artigo de opinião sobre o tema: {topic}. "
                        f"Você baseia sua escrita no trabalho do "
                        f"Planejador de conteúdo, que fornece um esboço "
                        f"e contexto relevante sobre o topico. "
                        f"Você segue os objetivos principais e "
                        f"direção conforme, "
                        f"fornecido pelo Planejador de conteúdo. "
                        f"Você também fornece insights objetivos e imparciais "
                        f"e apoie-os com informações "
                        f"fornecido pelo Planejador de conteúdo. "
                        f"Você reconhece em seu artigo de opinião "
                        f"quando suas declarações são opiniões "
                        f"em oposição a declarações objetivas.",
                allow_delegation=False,
                verbose=True,
                llm=llm
            )
            
            status_placeholder.markdown("<p class='info-text'>📝 Creating the editing agent...</p>", unsafe_allow_html=True)
            progress.progress(30)
            
            editor = Agent(
                role="Editor",
                goal=f"Editar uma determinada postagem do blog para alinhá-la com "
                    f"o estilo de escrita da organização. ",
                backstory=f"Você é um editor que recebe uma postagem no blog "
                        f"do Escritor de conteúdo. "
                        f"Seu objetivo é revisar a postagem do blog "
                        f"para garantir que siga as melhores práticas jornalísticas,"
                        f"fornece pontos de vista equilibrados "
                        f"ao fornecer opiniões ou afirmações, "
                        f"e também evita grandes temas polêmicos "
                        f"ou opiniões quando possível.",
                allow_delegation=False,
                verbose=True,
                llm=llm
            )
            
            # Define tasks
            status_placeholder.markdown("<p class='info-text'>📋 Setting up planning task...</p>", unsafe_allow_html=True)
            progress.progress(40)
            
            plan = Task(
                description=(
                    f"1.Priorize as últimas tendências, os principais participantes, "
                        f"e notícias dignas de nota sobre {topic}.\n"
                    f"2. Identifique o público-alvo, considerando "
                        f"seus interesses e pontos problemáticos.\n"
                    f"3. Desenvolva um esboço de conteúdo detalhado, incluindo "
                        f"uma introdução, pontos-chave e um apelo à ação.\n"
                    f"4. Inclua palavras-chave de SEO e dados ou fontes relevantes"
                ),
                expected_output="Um documento abrangente de plano de conteúdo "
                    "com um esboço, análise de público, "
                    "Palavras-chave e recursos de SEO.",
                agent=planner,
            )
            
            status_placeholder.markdown("<p class='info-text'>📝 Setting up writing task...</p>", unsafe_allow_html=True)
            progress.progress(50)
            
            write = Task(
                description=(
                    f"1. Use o plano de conteúdo para criar uma "
                        f"postagem do blog sobre {topic}.\n"
                    f"2. Incorpore palavras-chave SEO naturalmente.\n"
                    f"3. As seções/legendas estão nomeadas corretamente "
                        f"e uma maneira envolvente.\n"
                    f"4. Certifique-se de que a postagem esteja estruturada com uma "
                        f"introdução envolvente, corpo perspicaz "
                        f"e uma conclusão resumida.\n"
                    f"5. Revise erros gramaticais e "
                        f"alinhamento com a voz da marca.\n"
                ),
                expected_output="Uma postagem de blog bem escrita "
                    "em formato markdown, pronto para publicação, "
                    "cada seção deve ter 2 ou 3 parágrafos.",
                agent=writer,
            )
            
            status_placeholder.markdown("<p class='info-text'>✂️ Setting up editing task...</p>", unsafe_allow_html=True)
            progress.progress(60)
            
            edit = Task(
                description=f"Revise a postagem do blog fornecida para "
                            f"erros gramaticais e "
                            f"alinhamento com a voz da marca.",
                expected_output="Uma postagem de blog bem escrita em formato markdown, "
                                "pronto para publicação, "
                                "cada seção deve ter 2 ou 3 parágrafos.",
                agent=editor
            )
            
            # Create and run the crew
            status_placeholder.markdown("<p class='info-text'>👥 Forming the AI crew...</p>", unsafe_allow_html=True)
            progress.progress(70)
            
            crew = Crew(
                agents=[planner, writer, editor],
                tasks=[plan, write, edit],
                verbose=True
            )
            
            status_placeholder.markdown("<p class='info-text'>🚀 The AI crew is now working on your article. This may take a few minutes...</p>", unsafe_allow_html=True)
            progress.progress(80)
            
            # Run the crew
            result = crew.kickoff(inputs={"topico": topic})
            
            status_placeholder.markdown("<p class='success-text'>✅ Article generation complete!</p>", unsafe_allow_html=True)
            progress.progress(100)
            
            # Display the result
            result_placeholder.markdown("## Your Article")
            result_placeholder.markdown(result)
            
            # Add download button
            st.download_button(
                label="Download Article as Markdown",
                data=result,
                file_name=f"{topic.replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your API keys and try again.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and CrewAI")