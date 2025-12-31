# پروژه: تشخیص علائم راهنمایی و رانندگی (Traffic Sign Recognition)

**پیاده‌سازی:** شبکه عصبی کانولوشنی (CNN)  
**دیتاست:** GTSRB (German Traffic Sign Recognition Benchmark)  
**محیط اجرا:** Google Colab (پیشنهاد شده) / محلی (با GPU توصیه می‌شود)  
**نویسنده:** [محمد امین حری فراهانی]  
**لینک Colab:** https://colab.research.google.com/drive/1GYeo1AGNb_jn4WH3WBA0XvUCpc26DPmC?usp=sharing

---

## خلاصهٔ پروژه
این پروژه یک سامانهٔ طبقه‌بندی تصاویر علائم راهنمایی و رانندگی است که با استفاده از یک شبکهٔ عصبی کانولوشنی ساده (دو بلوک کانولوشن + لایه‌های Dense) و دیتاست GTSRB پیاده‌سازی شده است. مدل برای ۴۳ کلاس آموزش داده شده و نتایج آموزش/اعتبارسنجی، نمودارها و ماتریس درهم‌ریختگی در گزارش و نوت‌بوک ارائه شده‌اند.

---

## ساختار مخزن (نمونه)
.
├── README.md
├── report.tex # فایل لاتک گزارش (قابل کامپایل با XeLaTeX)
├── notebook.ipynb # نوت‌بوک اصلی (Google Colab)
├── src/ # کدهای پایتون (اختیاری)
│ ├── model.py
│ └── utils.py
├── data/ # مسیر پیشنهادی برای دیتا (غایب: دانلود از Kaggle)
├── figures/ # نمودارها و تصاویر خروجی
├── models/ # مدل ذخیره‌شده (.h5)
└── requirements.txt

yaml
Copy code
