import openai
from .prompts import CHATBOT_PROMPT, EXTRACTOR_PROMPT, MANAGER_PROMPT, UPDATE_PROMPT

openai.api_key = 'YOUR_API_KEY'


# 챗봇 에이전트
def chatbot_agent(user_input, conversation_history, missing_info=None):
    """
    챗봇 에이전트: 자연스러운 대화 진행
    """
    system_prompt = CHATBOT_PROMPT
    if missing_info:
        prompt_extension = f"현재 대화에서 다음 정보가 부족합니다: {', '.join(missing_info)}. 이를 자연스럽게 대화에 녹여서 부족한 정보를 수집하세요."
        system_prompt += prompt_extension

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )

    assistant_reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # 추천 가능 여부 판단
    if "추천" in assistant_reply:
        return assistant_reply, conversation_history, True

    return assistant_reply, conversation_history, False




# 추출 에이전트
def extractor_agent(conversation_history, contact_frequencies, hobbies, mbtis, ages):
    """
    대화 내용에서 정보를 추출하는 에이전트.
    취미(hobbies)도 추출해 업데이트합니다.
    """
    # 대화 텍스트 처리
    conversation_text = "\n".join(
        [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    )

    messages = [
        {"role": "system", "content": EXTRACTOR_PROMPT},
        {"role": "user", "content": conversation_text}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5
    )

    extracted_info = response.choices[0].message.content.strip()

    # 취미 추출
    if "연락빈도" in extracted_info:
        for contact_frequency in extracted_info.split("연락빈도:")[1].split(","):
            contact_frequencies.add(contact_frequency.strip())
    if "취미" in extracted_info:
        for hobby in extracted_info.split("취미:")[1].split(","):
            hobbies.add(hobby.strip())
    if "MBTI" in extracted_info:
        for mbti in extracted_info.split("MBTI:")[1].split(","):
            mbtis.add(mbti.strip())
    if "나이" in extracted_info:
        for age in extracted_info.split("나이:")[1].split(","):
            ages.add(age.strip())

    return {"contactFrequency": contact_frequencies, "hobby": hobbies, "mbti": mbtis, "age": ages}, extracted_info


def manager_agent(extracted_info, user_profile, extractor_message):
    """
    추출된 정보를 검토하고 부족한 정보를 관리하는 에이전트.
    """
    current_profile = {k: v if v else "정보 없음" for k, v in user_profile.items()}
    messages = [
        {"role": "system", "content": MANAGER_PROMPT},
        {"role": "user", "content": f"현재 프로필: {current_profile}\n추출된 정보: {extracted_info}\n추출 모델 메시지: {extractor_message}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5
    )

    evaluation = response.choices[0].message.content.strip()
    print(f"\n[평가 결과]:\n{evaluation}")

    # 프로필 업데이트 로직
    lines = evaluation.split("\n")[-5:]
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if "연락 빈도" in key:
            user_profile['contactFrequency'] = value
        elif "취미" in key:
            user_profile['hobby'] = value
        elif "MBTI" in key:
            user_profile['mbtiPreference'] = value
        elif "나이 선호" in key or "나이" in key:
            user_profile['agePreference'] = value

    # 부족한 정보 판별
    need_more_info = any(
        user_profile.get(key) in [None, "정보 없음"] for key in
        ['contactFrequency', 'hobby', 'mbtiPreference', 'agePreference']
    )

    return user_profile, need_more_info


def update_profile(user_profile, user_input):
    """
    사용자의 입력을 받아 프로필을 업데이트합니다.
    """
    change_flag = False

    current_profile = {k: v if v else "정보 없음" for k, v in user_profile.items()}
    messages = [
        {"role": "system", "content": UPDATE_PROMPT},
        {"role": "user", "content": f"현재 프로필: {current_profile}\n사용자 요청: {user_input}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5
    )
    evaluation = response.choices[0].message.content.strip()

    lines = evaluation.split("\n")[-5:]
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if "연락 빈도" in key:
            user_profile['contactFrequency'] = value
            change_flag = True
        elif "취미" in key:
            user_profile['hobby'] = value
            change_flag = True
        elif "MBTI" in key:
            user_profile['mbtiPreference'] = value
            change_flag = True
        elif "나이 선호" in key or "나이" in key:
            user_profile['agePreference'] = value
            change_flag = True

    return user_profile, change_flag