import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 피부 상태 관련 질문과 답변 데이터
questions = [
    "피부 측정이 가능한가요? 어떻게 하면 되나요?",
    "피부 수분 상태가 어떤가요?",
    "주름이 어느 정도 있나요?",
    "모공 상태를 확인할 수 있나요?",
    "피부 탄력 상태가 궁금해요.",
    "피부 상태를 개선하려면 어떻게 해야 하나요?",
    "추천하는 스킨케어 제품이 있나요?",
    "현재 피부 상태를 분석해 주세요."
]

answers = [
    "피부 측정은 세수를 하신 맨 얼굴로 사진을 찍어 업로드 해주세요",
    "피부 수분 상태는 정상 범위로, 수분 함량이 충분합니다.",
    "피부에 미세한 주름이 보입니다. 주름 개선을 위한 제품 사용을 추천합니다.",
    "모공 상태는 평균적으로 양호하지만, 일부 부위에서 확대된 모공이 관찰됩니다.",
    "피부 탄력은 평균 이상으로, 탄력이 좋은 상태입니다.",
    "피부 수분을 높이기 위해서는 충분한 물 섭취와 보습제를 사용하세요.",
    "피부 타입에 맞는 보습제와 주름 개선 크림을 추천합니다.",
    "현재 얼굴 사진을 분석한 결과, 수분, 주름, 모공, 탄력 상태를 확인했습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})


# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("피부 상태 측정 챗봇")

# 이미지 표시
st.image("skin.png", caption="Welcome to the Restaurant Chatbot", use_column_width=True)

st.write("피부 상태에 관한 질문을 입력해보세요. 예: 피부 수분 상태가 어떤가요??")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
