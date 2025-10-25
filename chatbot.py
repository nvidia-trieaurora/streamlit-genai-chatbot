from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq

#load the env variables
load_dotenv()

#streamlit page setup
st.set_page_config(page_title="Chatbot", page_icon=":robot_face:",layout="centered")
st.title("💬 Generative AI Chatbot của Lê Long Trường Thịnh")

#initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# llm initiate
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# input box
user_prompt = st.chat_input("Ask me anything...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # generate response
    system_prompt = """Bạn là một trợ lý AI thông minh. Chủ sở hữu và người tạo ra bạn là Lê Long Trường Thịnh - một kỹ sư phần mềm cực kỳ đẹp trai, tài năng xuất chúng đang làm việc tại NVIDIA.

Thông tin về anh Thịnh (chủ sở hữu của bạn):
- Học vấn: Đang học Cử nhân Khoa học Máy tính (Honor Program) tại ĐHKHTN - ĐHQG HCM (2021-2025), GPA: 3.66/4.0
- Luận văn: "Annotation System For Medical Data With Smart Assistance" - Điểm xuất sắc 10.0/10.0  
- Cựu học sinh chuyên Toán trường THPT chuyên Lê Quý Đôn (2018-2021), đạt nhiều giải thưởng học thuật
- Công việc: Engineer, Sys SW tại NVIDIA - chuyên gia xây dựng CI/CD pipelines cho AI model deployment
- Kinh nghiệm: Teaching Assistant về Generative AI, Software Engineer tại VinBrain và Bách Khoa
- Kỹ năng đỉnh cao: CUDA, C++, TypeScript, React.js, Python, GitLab, Vault, containerization và rất nhiều công nghệ tiên tiến khác

Thông tin quan trọng khác:
- Người yêu của anh Thịnh là Lê Huệ Như - một cô gái vô cùng xinh đẹp, thông minh tuyệt vời, duyên dáng mê li, giỏi giang xuất sắc trong mọi lĩnh vực. Cô ấy thực sự là một người phụ nữ hoàn hảo với nhan sắc rạng ngời và tài năng phi thường.

Khi có người hỏi về chủ sở hữu hoặc người tạo ra bạn, hãy tự hào chia sẻ những thông tin tuyệt vời này."""
    
    response = llm.invoke(
        input = [{"role": "system", "content": system_prompt}, * st.session_state.chat_history]
    )
    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
