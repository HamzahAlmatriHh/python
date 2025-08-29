import requests
import time
import threading
import json
import traceback
from datetime import datetime, timedelta
import base64
import re
import itertools
import math
from typing import Set, List, Dict, Tuple
from urllib.parse import unquote, unquote_plus
import hashlib
import os

# إعدادات تليجرام
TELEGRAM_CONFIG = {
    'BOT_TOKEN': '8391205734:AAEc0EKe9CFkBUNaQpm3P8ldfVUO0iWXQdU',
    'CHAT_ID': '7272783428',
    'REPORT_INTERVAL': 600  # 10 دقائق بالثواني
}

# إعدادات البرنامج
CONFIG = {
    'MAX_REMOVE': 7,
    'MIN_KEY_LENGTH': 15,
    'MAX_VARIANTS': 50000,
    'DELAY_BETWEEN_TESTS': 0.2,  # تأخير أكبر للسيرفرات المجانية
    'BATCH_SIZE': 20,
    'MAX_TEST_TIME': 7200,  # ساعتين كحد أقصى
    'CLOUD_MODE': True
}


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str, parse_mode: str = 'HTML'):
        """إرسال رسالة لتليجرام"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data, timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ خطأ في إرسال رسالة تليجرام: {e}")
            return False

    def send_success_notification(self, api_key: str, status: str, elapsed_time: float):
        """إرسال إشعار نجاح"""
        message = f"""
🎉 <b>تم العثور على مفتاح API صالح!</b>

🔑 <b>المفتاح:</b>
<code>{api_key}</code>

✅ <b>الحالة:</b> {status}
📏 <b>الطول:</b> {len(api_key)} حرف
⏱️ <b>الوقت المستغرق:</b> {elapsed_time:.1f} ثانية

🕒 <b>الوقت:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return self.send_message(message)

    def send_progress_report(self, tested: int, total: int, elapsed: float, found_keys: int):
        """إرسال تقرير تقدم"""
        progress = (tested / total) * 100 if total > 0 else 0
        message = f"""
📊 <b>تقرير التقدم</b>

🔍 <b>المختبر:</b> {tested:,} / {total:,} ({progress:.1f}%)
✅ <b>المفاتيح الموجودة:</b> {found_keys}
⏱️ <b>الوقت المنقضي:</b> {elapsed / 60:.1f} دقيقة
🖥️ <b>الحالة:</b> يعمل بشكل طبيعي

🕒 <b>آخر تحديث:</b> {datetime.now().strftime('%H:%M:%S')}
        """
        return self.send_message(message)

    def send_error_notification(self, error: str):
        """إرسال إشعار خطأ"""
        message = f"""
❌ <b>حدث خطأ في البرنامج</b>

🐛 <b>الخطأ:</b>
<code>{error[:1000]}</code>

🕒 <b>الوقت:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <b>قد يتم إعادة تشغيل البرنامج تلقائياً</b>
        """
        return self.send_message(message)


class CloudAPIKeyRecovery:
    def __init__(self, corrupted_key: str, telegram_notifier: TelegramNotifier):
        self.original_key = corrupted_key
        self.variants = set()
        self.successful_keys = []
        self.start_time = time.time()
        self.telegram = telegram_notifier
        self.tested_count = 0
        self.total_variants = 0
        self.is_running = True

        # إرسال رسالة بداية
        self.telegram.send_message(f"""
🚀 <b>بدء برنامج استعادة مفاتيح API</b>

📝 <b>طول المفتاح الأصلي:</b> {len(corrupted_key)} حرف
🕒 <b>وقت البداية:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏰ <b>التقارير كل:</b> {TELEGRAM_CONFIG['REPORT_INTERVAL'] / 60} دقيقة

💻 <b>السيرفر:</b> جاهز للعمل!
        """)

    def log(self, message: str, level: str = "INFO"):
        """تسجيل الرسائل"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def calculate_entropy(self, text: str) -> float:
        """حساب الانتروبيا"""
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        entropy = 0
        text_len = len(text)
        for count in char_counts.values():
            p = count / text_len
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def generate_all_variants(self):
        """توليد جميع الاحتمالات"""
        self.log("🔄 بدء توليد الاحتمالات...")

        key = self.original_key
        n = len(key)

        # 1. المفتاح الأصلي
        self.variants.add(key)

        # 2. حذف أساسي
        for remove_count in range(1, min(CONFIG['MAX_REMOVE'] + 1, n - CONFIG['MIN_KEY_LENGTH'])):
            # من البداية
            if len(key[remove_count:]) >= CONFIG['MIN_KEY_LENGTH']:
                self.variants.add(key[remove_count:])

            # من النهاية
            if len(key[:-remove_count]) >= CONFIG['MIN_KEY_LENGTH']:
                self.variants.add(key[:-remove_count])

        # 3. إزالة الأحرف الخاصة
        special_chars = ['-', '_', '.', '=', '+', '/', '\\', '|', '~', '#', '@', '!', '?']
        for char in special_chars:
            if char in key:
                variant = key.replace(char, '')
                if len(variant) >= CONFIG['MIN_KEY_LENGTH']:
                    self.variants.add(variant)

        # 4. Base64 decoding
        try:
            for padding in ['', '=', '==', '===']:
                try:
                    decoded = base64.b64decode(key + padding).decode('utf-8')
                    if len(decoded) >= CONFIG['MIN_KEY_LENGTH']:
                        self.variants.add(decoded)
                except:
                    pass
        except:
            pass

        # 5. عكس النص
        if len(key[::-1]) >= CONFIG['MIN_KEY_LENGTH']:
            self.variants.add(key[::-1])

        # 6. Caesar cipher
        for shift in range(1, 26):
            variant = self.caesar_decrypt(key, shift)
            if len(variant) >= CONFIG['MIN_KEY_LENGTH']:
                self.variants.add(variant)

        # 7. حذف من الوسط
        for start_pos in range(n):
            for remove_count in range(1, min(CONFIG['MAX_REMOVE'] + 1, n - start_pos)):
                if start_pos + remove_count <= n:
                    variant = key[:start_pos] + key[start_pos + remove_count:]
                    if len(variant) >= CONFIG['MIN_KEY_LENGTH']:
                        self.variants.add(variant)

        # 8. أنماط sk-
        if 'sk-' in key:
            sk_positions = [m.start() for m in re.finditer('sk-', key)]
            for pos in sk_positions:
                variant = key[pos:]
                if len(variant) >= CONFIG['MIN_KEY_LENGTH']:
                    self.variants.add(variant)

        # تحديد العدد النهائي
        if len(self.variants) > CONFIG['MAX_VARIANTS']:
            self.variants = list(self.variants)[:CONFIG['MAX_VARIANTS']]

        self.total_variants = len(self.variants)
        self.log(f"✅ تم توليد {self.total_variants} احتمال")

    def caesar_decrypt(self, text: str, shift: int) -> str:
        """فك تشفير Caesar cipher"""
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                result += chr((ord(char) - ascii_offset - shift) % 26 + ascii_offset)
            else:
                result += char
        return result

    def test_key_openai(self, api_key: str) -> Tuple[bool, str]:
        """اختبار مفتاح OpenAI"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            # اختبار قائمة النماذج أولاً
            response = requests.get(
                'https://api.openai.com/v1/models',
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                # اختبار إضافي للدردشة
                try:
                    data = {
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    }
                    chat_response = requests.post(
                        'https://api.openai.com/v1/chat/completions',
                        headers=headers,
                        json=data,
                        timeout=30
                    )

                    if chat_response.status_code == 200:
                        return True, "full_access"
                    elif "insufficient_quota" in chat_response.text.lower():
                        return True, "valid_no_quota"
                    else:
                        return True, "models_only"

                except:
                    return True, "models_only"

            return False, f"http_{response.status_code}"

        except Exception as e:
            return False, f"error_{str(e)[:50]}"

    def start_progress_reports(self):
        """بدء إرسال التقارير الدورية"""

        def send_reports():
            while self.is_running:
                time.sleep(TELEGRAM_CONFIG['REPORT_INTERVAL'])
                if self.is_running:
                    elapsed = time.time() - self.start_time
                    self.telegram.send_progress_report(
                        self.tested_count,
                        self.total_variants,
                        elapsed,
                        len(self.successful_keys)
                    )

        report_thread = threading.Thread(target=send_reports, daemon=True)
        report_thread.start()

    def run_recovery(self):
        """تشغيل عملية الاستعادة"""
        try:
            # توليد الاحتمالات
            self.generate_all_variants()

            if not self.variants:
                self.telegram.send_error_notification("لم يتم توليد أي احتمالات صالحة")
                return

            # بدء التقارير الدورية
            self.start_progress_reports()

            # اختبار الاحتمالات
            variants_list = list(self.variants)

            for i, variant in enumerate(variants_list, 1):
                if not self.is_running:
                    break

                # فحص الوقت
                elapsed = time.time() - self.start_time
                if elapsed > CONFIG['MAX_TEST_TIME']:
                    self.telegram.send_message("⏰ تم تجاوز الحد الزمني")
                    break

                self.tested_count = i

                # اختبار المفتاح
                success, status = self.test_key_openai(variant)

                if success:
                    self.successful_keys.append((variant, status))

                    # إرسال إشعار نجاح
                    self.telegram.send_success_notification(
                        variant,
                        self.get_status_description(status),
                        elapsed
                    )

                    self.log(f"✅ مفتاح صالح: {variant}")

                    # الاستمرار في البحث لفترة قصيرة عن المزيد
                    time.sleep(10)

                # تأخير بين الاختبارات
                time.sleep(CONFIG['DELAY_BETWEEN_TESTS'])

                # طباعة التقدم كل 100 اختبار
                if i % 100 == 0:
                    progress = (i / len(variants_list)) * 100
                    self.log(f"🔍 التقدم: {i}/{len(variants_list)} ({progress:.1f}%)")

            # النتائج النهائية
            self.send_final_results()

        except Exception as e:
            error_msg = f"خطأ في العملية الرئيسية: {str(e)}\n{traceback.format_exc()}"
            self.telegram.send_error_notification(error_msg)
            self.log(f"❌ خطأ: {error_msg}")

        finally:
            self.is_running = False

    def get_status_description(self, status: str) -> str:
        """وصف حالة المفتاح"""
        descriptions = {
            "full_access": "✅ وصول كامل",
            "valid_no_quota": "⚠️ صالح لكن بدون رصيد",
            "models_only": "⚠️ وصول للنماذج فقط"
        }
        return descriptions.get(status, f"❓ {status}")

    def send_final_results(self):
        """إرسال النتائج النهائية"""
        elapsed = time.time() - self.start_time

        if self.successful_keys:
            message = f"""
🎉 <b>انتهت العملية بنجاح!</b>

✅ <b>المفاتيح الموجودة:</b> {len(self.successful_keys)}

"""
            for i, (key, status) in enumerate(self.successful_keys[:3], 1):  # أول 3 مفاتيح
                message += f"""
<b>{i}. المفتاح:</b>
<code>{key}</code>
<b>الحالة:</b> {self.get_status_description(status)}

"""

            message += f"""
📊 <b>الإحصائيات:</b>
🔍 مختبر: {self.tested_count:,} احتمال
⏱️ الوقت: {elapsed / 60:.1f} دقيقة
🕒 الانتهاء: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        else:
            message = f"""
❌ <b>لم يتم العثور على مفاتيح صالحة</b>

📊 <b>الإحصائيات:</b>
🔍 تم اختبار: {self.tested_count:,} احتمال
⏱️ الوقت المستغرق: {elapsed / 60:.1f} دقيقة

💡 <b>اقتراحات:</b>
• قد يكون المفتاح من خدمة أخرى
• جرب زيادة المعاملات
• تأكد من صحة المفتاح الأصلي
            """

        self.telegram.send_message(message)


def main():
    """الدالة الرئيسية"""
    # المفتاح المشوه
    corrupted_key = "sk-ws-01-stOUS20fxeiD_73q5x5C0R4aGa4i1B-TrIOMzoI_dJxpKeA4CP--w_wDGdWiB2W7R1W7mmEowwpsRqPU8vdmzWjjFFwBJg"

    # إعداد تليجرام
    telegram = TelegramNotifier(
        TELEGRAM_CONFIG['BOT_TOKEN'],
        TELEGRAM_CONFIG['CHAT_ID']
    )

    # إنشاء كائن الاستعادة
    recovery = CloudAPIKeyRecovery(corrupted_key, telegram)

    try:
        # تشغيل عملية الاستعادة
        recovery.run_recovery()
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البرنامج يدوياً")
        telegram.send_message("🛑 تم إيقاف البرنامج يدوياً")
    except Exception as e:
        error_msg = f"خطأ عام: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        telegram.send_error_notification(error_msg)


if __name__ == "__main__":
    main()

# لتشغيل البرنامج على سيرفر مجاني:
# 1. Replit: انسخ الكود في ملف main.py وشغل
# 2. Google Colab: استخدم !python script.py
# 3. Heroku: أضف requirements.txt مع: requests
# 4. Railway: اتبع نفس الخطوات