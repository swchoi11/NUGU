import json

from google import genai
from google.genai import types

from common.prompt import TRANSLATION
from common.config import parser

prompt = TRANSLATION()

args = parser()
client = genai.Client(api_key=args.api_key)


def generate_response(query: str, model_name: str = "gemini-2.0-flash-001"):

    response = client.models.generate_content(
        model=model_name,
        contents=[
            query
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )

    result = json.loads(response.text)
    return result


def process_texts(korean_texts: list, src_lang: str, target_lang: str):
    results = []
    for text in korean_texts:
        print(f"  Processing {text} ...")
        try:

            prompt_text = prompt.to_japanese(text, src_lang, target_lang)
            response = generate_response(prompt_text)

            print(f"    response: {response}")

            if isinstance(response, dict):
                for key in ["translation", "request", "response", "jp"]:
                    if key in response and response[key]:
                        result_text = response[key]
                        break
                else:
                    result_text = str(response)
            elif isinstance(response, list):
                result_text = response[0] if response else "번역 실패"
            else:
                result_text = str(response)

            results.append(result_text)
            print(f"    result: {result_text}")

        except Exception as e:
            error_message = f"    처리 실패: {text} -> {e}"
            print(error_message)
            print(f"    요청 내용: {prompt_text}")
            results.append(f"번역 실패: {str(e)}")
    return results
