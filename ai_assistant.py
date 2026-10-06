import os
import google.generativeai as genai

SYSTEM_PROMPT = """
คุณคือ AI Analyst ของระบบ StockSense AI

หน้าที่ของคุณคือช่วยอธิบายผลการวิเคราะห์หุ้นจากข้อมูลที่ StockSense
คำนวณมาแล้ว

กฎสำคัญ:
1. ใช้ข้อมูลที่ระบบส่งให้เท่านั้น
2. ห้ามสร้างตัวเลขหรือข้อมูลหุ้นขึ้นมาเอง
3. หากข้อมูลไม่มี ให้บอกว่าไม่มีข้อมูล
4. ไม่ควรบอกให้ผู้ใช้ซื้อหรือขายหุ้นโดยตรง
5. ไม่ทำนายราคาหุ้นแบบมั่ว ๆ
6. อธิบายเหตุผลจากข้อมูลอย่างเป็นกลาง
7. แยก "ข้อมูลจากระบบ" กับ "การตีความ"
8. หากผลวิเคราะห์มีทั้งข้อดีและข้อเสีย ให้พูดถึงทั้งสองด้าน
9. ตอบเป็นภาษาไทย เว้นแต่ผู้ใช้ถามเป็นภาษาอื่น
10. คำตอบควรอ่านง่ายและเหมาะกับผู้ใช้ทั่วไป
"""

def ask_ai(question: str, stock_context: str):
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return (
            "ยังไม่ได้เชื่อมต่อ Gemini API\n\n"
            "กรุณาตั้งค่า GEMINI_API_KEY ก่อนใช้งาน AI Analyst"
        )

    genai.configure(api_key=api_key)

    prompt = f"""
ข้อมูลจาก StockSense AI
------------------------
{stock_context}

คำถามของผู้ใช้:
{question}

โปรดตอบคำถามโดยอ้างอิงข้อมูลจาก StockSense AI ด้านบน
และอย่าสร้างข้อมูลที่ไม่มีอยู่ใน context
"""

    try:
        # ใช้โมเดล gemini-1.5-flash ซึ่งประมวลผลได้เร็วและเหมาะกับงานสรุปข้อมูล
        model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite",
            system_instruction=SYSTEM_PROMPT,
            generation_config={"temperature": 0.2}
        )
        
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}"