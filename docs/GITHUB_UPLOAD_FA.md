# راهنمای جایگزینی کامل مخزن GitHub

این ZIP «محتویات مخزن» است؛ بارگذاری یک ZIP در صفحهٔ GitHub باعث بازشدن خودکار پوشه‌های آن نمی‌شود. ZIP را استخراج کنید و محتویات پوشه `CIRG-PoValley-PM25` را در شاخهٔ اصلی مخزن قرار دهید.

## روش پیشنهادی با Git

اگر مخزن فعلی خالی است، در رایانه در پوشهٔ استخراج‌شده دستورهای زیر را اجرا کنید (URL مخزن مقصد خود را در خط `remote add` بنویسید):

```bash
git init
git branch -M main
git add .
git commit -m "Restore CIRG reproducibility repository and EMA manuscript"
git remote add origin https://github.com/OWNER/CIRG-PoValley-PM25.git
git push -u origin main
```

اگر مخزن قبلی دارای تاریخچه و تعارض است، بدون بررسی تاریخچه از `push --force` استفاده نکنید. می‌توانید پوشه‌ها را در یک clone از مخزن موجود جایگزین کنید و commit جدید بسازید.

## داده‌های بزرگ

آرشیو آفلاینِ مستقل، فایل خام ۲۵۲ مگابایتی و پردازش‌شدهٔ ۵۹ مگابایتی را دارد. این دو فایل در بستهٔ انتشار عمومی GitHub وجود ندارند، چون فایل خام بالاتر از سقف فایل معمولی GitHub است و شرایط بازنشر منبع اولیه نیز باید رعایت شود. پس از دریافت اجازهٔ لازم، در صورت تمایل از Git LFS استفاده کنید:

```bash
git lfs install
git lfs track "data/raw/*.csv"
git lfs track "data/processed/*.csv"
git add .gitattributes
git add -f data/raw/PoValleyPM2520132023csv.csv data/processed/PoValleyPM25Final20182023Observed.csv
git commit -m "Add approved data via Git LFS"
git push
```

برای صرفاً بازتولید، نیازی به انتشار فایل‌های حجیم در GitHub نیست: دادهٔ منبع از Zenodo قابل تهیه است و بستهٔ آفلاین خصوصیِ همین پروژه ورودی‌های دقیق را دارد.
