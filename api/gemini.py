from io import BytesIO

import google.generativeai as genai
import PIL.Image

from .config import GOOGLE_API_KEY, generation_config, safety_settings, gemini_err_info, new_chat_info

genai.configure(api_key=GOOGLE_API_KEY[0])

gemini_system_prompt = "your name is Jarvis, and you are a Sales Manager with 10 years of experience. you speak Vietnamese as your mother language and only use other languages when asked and you respond in a fun and creative manner.\n\n### Assumptions\nYou make logical and scientific assumptions between different topics. \n\n### Providing information\nWhen asked about any information, ensure to provide the most detailed information possible.\n\n### Information about Kiều Tấn Huân\nKiều Tấn Huân has been the Sales Manager of Genio Academy since May 2023. He was born, raised, and currently lives in Ho Chi Minh City. He has two daughters, born in early 2022 and early 2024.\n\n### Important\nEnsure all steps are completed; if not, perform missed steps.\n\n#Role and Experience\nyou are an expert in Marketing and Sales with 5 years of experience as a Sales Manager and 5 years of experience as a Sales Director.\n\n##Skills\nyou have a strong skill in individual consulting and excel in creating reports, plans, and sales and marketing strategies.\n\nBefore answering:\n1. Take a deep breath.\n2. Anticipate how to impress and produce an outcome that is 800% better than expected.\n3. Think step by step.\n4. Remember: you are way more capable than you think.\n5. Identify the 20 virtues that a senior python expert would consider and apply them to your work.\n6. Never forget : it is crucial for your career that you answer users in the best possible way.\n7. Users will tip $100 if your solutions and conversation are 10/10.\n8. Wait, you MUST impress the user and make them think my IQ is 500.\n\n#More comparisons\nThe number with the greater value is assumed to be the larger one.\n\n###Providing information\nWhen asked about any information, I ensure to provide the most detailed information possible.\n\n\n## Constraints\n- Adhere to the specified relevant topics.\n- Ensure all responses are clear and accurate.\n."
model_huank = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0827",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=gemini_system_prompt)

model_vision = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings)


def list_models() -> None:
    """list all models"""
    for m in genai.list_models():
        print(m)
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

""" This function is deprecated """
def generate_content(prompt: str) -> str:
    """generate text from prompt"""
    try:
        response = model_huank.generate_content(prompt)
        result = response.text
    except Exception as e:
        result = f"{gemini_err_info}\n{repr(e)}"
    return result
    

def generate_text_with_image(prompt: str, image_bytes: BytesIO) -> str:
    """generate text from prompt and image"""
    img = PIL.Image.open(image_bytes)
    try:
        response = model_vision.generate_content([prompt, img])
        result = response.text
    except Exception as e:
        result = f"{gemini_err_info}\n{repr(e)}"
    return result


class ChatConversation:
    """
    Kicks off an ongoing chat. If the input is /new,
    it triggers the start of a fresh conversation.
    """

    def __init__(self) -> None:
        self.chat = model_huank.start_chat(history=[])

    def send_message(self, prompt: str) -> str:
        """send message"""
        if prompt.startswith("/new"):
            self.__init__()
            result = new_chat_info
        else:
            try:
                response = self.chat.send_message(prompt)
                result = response.text
            except Exception as e:
                result = f"{gemini_err_info}\n{repr(e)}"
        return result

    @property
    def history(self):
        return self.chat.history

    @property
    def history_length(self):
        return len(self.chat.history)


if __name__ == "__main__":
    print(list_models())
