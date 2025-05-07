import os
import time
import json
import base64
import pandas as pd

from common.prompt import TRANSLATION
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import Content, Part, SafetySetting, GenerationConfig

prompt = TRANSLATION()

# {text} 주어진 문장을 한국어에서 일본어로 정확하게 번역해주세요.
# 문법적 자연스러움과 의미 보존을 우선시하며, 번역된 일본어 텍스트만 반환해주세요.
# 불필요한 설명, 주석, 인용부호는 포함하지 마세요.


# def get_prompt(text):
#     prompt = f"""
#     {text}
#     Translate the following sentence from Korean to Japanese with high accuracy.
#     Preserve grammatical correctness and semantic fidelity.
#     Return only the translated Japanese text. No explanations, quotes, or formatting.
#     """
#     return prompt


def generate_response(query: str, model_name: str = "gemini-2.0-flash-001") -> str:
    contents = [
        Content(
            role="user",
            parts=[Part.from_text(query)]
        )
    ]

    safety_settings = [
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]

    generate_content_config = GenerationConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=128,
        response_mime_type="application/json",
    )
    model = GenerativeModel(
        model_name=model_name,
        generation_config=generate_content_config,
        safety_settings=safety_settings
    )

    response = model.generate_content(contents)
    time.sleep(2)
    return response.text


def process_texts(korean_texts, src_lang, target_lang):
    rows = []
    for text in korean_texts:
        print(f"  Processing {text} ...")
        try:
            response = generate_response(prompt.to_japanese(text, src_lang, target_lang))
            # response = json.loads(response)
            rows.append(response)
            print(f"    row: {rows}")
        except Exception as e:
            print(f"    처리 실패: {text} -> {e}")
            continue
    return pd.DataFrame(rows)


# if __name__ == "__main__":

#     korean_texts = [
#         "자연주름/힘줄",
#         "미세스크래치가 있습니다"
#     ]

#     df = process_texts(korean_texts)

#     print(df)
