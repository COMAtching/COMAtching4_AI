import gradio as gr
import pandas as pd
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from src.agent import chatbot_agent, extractor_agent, manager_agent, update_profile
from src.functions import recommendation_function, deliver_recommendation

# Initialize embedding model globally
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj


def login(user_id):
    df = pd.read_excel('data.xlsx')

    # 사용자 ID가 데이터셋에 있는지 확인
    if user_id not in df['id'].astype(str).values:
        return (
            "Invalid user id. Please try again.",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            None, None, None, None, None, None
        )
    else:
        # 사용자 데이터 가져오기
        user_data = df[df['id'].astype(str) == user_id].iloc[0]

        # 사용자 프로필 생성
        user_profile = {
            'id': str(user_data['id']),
            'age': int(user_data['age']) if pd.notnull(user_data['age']) else None,
            'gender': user_data['gender'] if pd.notnull(user_data['gender']) else None,
            'contactFrequency': None,  # Default value for dynamic update
            'hobby': None,  # Default value for dynamic update
            'mbtiPreference': user_data.get('mbti', None) if pd.notnull(user_data.get('mbti', None)) else None,
            'agePreference': None,  # Default value for dynamic update
            'major': user_data['major'] if pd.notnull(user_data['major']) else None,
        }

        # 누락된 값 확인 및 출력
        missing_fields = [key for key, value in user_profile.items() if value is None]
        if missing_fields:
            print(f"Warning: The following fields are missing for user {user_id}: {missing_fields}")

        # 기존 대화 이력 로드
        conversation_file = f'conversation_history_{user_id}.json'
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r') as f:
                data = json.load(f)
                conversation_history = data.get('conversation_history', [])
                user_profile.update(data.get('user_profile', {}))
                phase = data.get('phase', 'chat')
        else:
            # 새로운 대화 시작
            conversation_history = []
            chatbot_first_message = "안녕! 요즘 어떻게 지내? 특별한 일이나 재미있는 취미거리가 있었는지 궁금해. 공유해줄 수 있어?"
            conversation_history.append({"role": "assistant", "content": chatbot_first_message})
            phase = 'chat'

        # 대화 메시지 준비
        messages = [
            ("당신", msg['content']) if msg['role'] == 'user' else ("챗봇", msg['content'])
            for msg in conversation_history
        ]

        # 단계별 UI 가시성 설정
        if phase == 'end':
            user_input_visibility = gr.update(visible=False)
            send_button_visibility = gr.update(visible=False)
        else:
            user_input_visibility = gr.update(visible=True)
            send_button_visibility = gr.update(visible=True)

        return (
            "",
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
            messages, user_profile, conversation_history, phase,
            user_input_visibility, send_button_visibility
        )


def chatbot_interact(user_input, conversation_history_state, user_profile_state, phase_state, user_id):
    # Get states
    conversation_history = conversation_history_state or []
    user_profile = user_profile_state or {}
    phase = phase_state or 'chat'

    if phase == 'chat':
        # Append user's message
        conversation_history.append({"role": "user", "content": user_input})
        # Count the number of user turns
        user_turns = len([msg for msg in conversation_history if msg['role'] == 'user'])

        if user_turns < 9:
            # Call chatbot_agent
            chatbot_reply, conversation_history, _ = chatbot_agent(
                user_input, conversation_history
            )
            # Append chatbot's response
            conversation_history.append({"role": "assistant", "content": chatbot_reply})
            # Prepare messages for display
            messages = []
            for msg in conversation_history:
                if msg['role'] == 'user':
                    messages.append(("당신", msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(("챗봇", msg['content']))
            return messages, conversation_history, user_profile, phase, gr.update(value=''), gr.update(visible=True)
        else:
            # Proceed to confirmation phase
            chatbot_reply = "이제 대화를 요약하고 적절한 사람을 찾아볼게!"
            conversation_history.append({"role": "assistant", "content": chatbot_reply})
            # Extract information and update user_profile
            extracted_info, extractor_message = extractor_agent(conversation_history, set(), set(), set(), set())
            user_profile, _ = manager_agent(extracted_info, user_profile, extractor_message)
            # Prepare confirmation message
            confirm_message = f"너의 이상형은 이런 사람인 것 같아!\n연락빈도: {user_profile.get('contactFrequency', '정보 없음')}, 취미: {user_profile.get('hobby', '정보 없음')}, MBTI: {user_profile.get('mbtiPreference', '정보 없음')}, 나이 선호: {user_profile.get('agePreference', '정보 없음')}\n혹시 내가 파악한 너의 이상형이 마음에 들어? 잘못된 부분이 있으면 말해줘!"
            conversation_history.append({"role": "assistant", "content": confirm_message})
            messages = []
            for msg in conversation_history:
                if msg['role'] == 'user':
                    messages.append(("당신", msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(("챗봇", msg['content']))
            # Set phase to 'confirm'
            phase = 'confirm'
            return messages, conversation_history, user_profile, phase, gr.update(value=''), gr.update(visible=True)
    elif phase == 'confirm':
        # User provides confirmation or corrections
        conversation_history.append({"role": "user", "content": user_input})
        # Update profile
        user_profile, change_flag = update_profile(user_profile, user_input)
        if change_flag:
            # Inform the user that the profile is updated
            chatbot_reply = f"프로필이 업데이트되었어! 다시 한번 확인해볼게!\n연락빈도: {user_profile.get('contactFrequency', '정보 없음')}, 취미: {user_profile.get('hobby', '정보 없음')}, MBTI: {user_profile.get('mbtiPreference', '정보 없음')}, 나이 선호: {user_profile.get('agePreference', '정보 없음')}"
            conversation_history.append({"role": "assistant", "content": chatbot_reply})
        else:
            chatbot_reply = "프로필에 변경이 없었어!"
            conversation_history.append({"role": "assistant", "content": chatbot_reply})
        # Set phase to 'recommend'
        phase = 'recommend'
        # Prepare messages
        messages = []
        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(("당신", msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(("챗봇", msg['content']))
        return messages, conversation_history, user_profile, phase, gr.update(value=''), gr.update(visible=True)
    elif phase == 'recommend':
        # Proceed to recommendation
        # Call recommendation_function
        recommended_person = recommendation_function(user_profile, 'data.xlsx', embedding_model, user_id)
        if recommended_person is not None:
            # Prepare recommendation message
            recommendation_message = deliver_recommendation(recommended_person, conversation_history)
            conversation_history.append({"role": "assistant", "content": recommendation_message})
        else:
            conversation_history.append({"role": "assistant", "content": "코매칭을 가입한 사람중에 너의 이상형과 비슷한 사람이 없는 것 같아.. 관리자에게 문의해줘!"})
        # Save conversation history and user_profile
        data = {
            'conversation_history': conversation_history,
            'user_profile': user_profile,
            'phase': 'end'
        }
        data = convert_numpy_types(data)
        conversation_file = f'conversation_history_{user_id}.json'
        with open(conversation_file, 'w') as f:
            json.dump(data, f)
        # Set phase to 'end'
        phase = 'end'
        # Prepare messages
        messages = []
        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(("당신", msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(("챗봇", msg['content']))
        return messages, conversation_history, user_profile, phase, gr.update(value=''), gr.update(visible=False)
    else:
        # Conversation has ended
        messages = []
        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(("당신", msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(("챗봇", msg['content']))
        return messages, conversation_history, user_profile, phase, gr.update(value=''), gr.update(visible=False)

with gr.Blocks() as demo:
    user_id_input = gr.Textbox(label="Enter your ID")
    login_button = gr.Button("Login")
    login_message = gr.Textbox(visible=False)
    # Hidden components for chatbot interface
    chatbot_interface = gr.Chatbot(visible=False)
    user_input = gr.Textbox(visible=False, placeholder="Type your message here", show_label=False)
    send_button = gr.Button("Send", visible=False)
    # State variables
    user_profile_state = gr.State()
    conversation_history_state = gr.State()
    phase_state = gr.State()

    # Login function
    login_button.click(
        login,
        inputs=user_id_input,
        outputs=[
            login_message, chatbot_interface, user_input, send_button,
            chatbot_interface, user_profile_state, conversation_history_state, phase_state,
            user_input, send_button
        ]
    )

    # Chatbot interaction
    user_input.submit(
        chatbot_interact,
        inputs=[user_input, conversation_history_state, user_profile_state, phase_state, user_id_input],
        outputs=[
            chatbot_interface, conversation_history_state, user_profile_state, phase_state,
            user_input, send_button
        ]
    )

    send_button.click(
        chatbot_interact,
        inputs=[user_input, conversation_history_state, user_profile_state, phase_state, user_id_input],
        outputs=[
            chatbot_interface, conversation_history_state, user_profile_state, phase_state,
            user_input, send_button
        ]
    )

demo.launch(share=True)
