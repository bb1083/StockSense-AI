# StockSense AI v2

พิมพ์แค่ชื่อหุ้น (หรือชื่อบริษัท) แล้วได้การวิเคราะห์ครบวงจร

## วิธีติดตั้ง
1. คัดลอกไฟล์ทั้งหมดในโฟลเดอร์นี้ไปวางในโฟลเดอร์โปรเจกต์เดิม (ทับ `app.py` เดิม; โฟลเดอร์ `data/` ยังอยู่เหมือนเดิม)
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

ถ้าเปิดแอปแล้วแท็บ Model Card แจ้งว่าโหลดโมเดลไม่ได้ (scikit-learn คนละเวอร์ชัน) ให้รัน `python train_universal_model.py` หนึ่งครั้ง (ใช้เวลาไม่ถึงนาที)

## ไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | หน้าเว็บ (8 แท็บ) |
| `analytics.py` | เอนจินวิเคราะห์ทั้งหมด (Technical, Risk, Fundamentals, Scorecard) ไม่พึ่ง Streamlit |
| `data.py` | ดึงข้อมูล Yahoo Finance + cache |
| `ui.py` | ธีม CSS และ component |
| `train_universal_model.py` | ฝึกโมเดลความผันผวนที่ใช้กับหุ้นทุกตัว |
| `universal_vol_model.joblib` | โมเดลที่ฝึกแล้ว (80 KB) |

## ใช้กับหุ้นอะไรได้บ้าง
หุ้นสหรัฐ (AAPL, NVDA) ได้ข้อมูลครบที่สุด หุ้นไทยใส่ `.BK` (PTT.BK) ได้ แต่ Yahoo มักมีงบการเงินไม่ครบ ระบบจะลดระดับความเชื่อมั่นและป้ายสรุปให้อัตโนมัติ
