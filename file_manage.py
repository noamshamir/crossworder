import os
import shutil

def weekday_sakamoto(y, m, d):
    # 0=Sunday, 1=Monday, ..., 6=Saturday
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    if m < 3:
        y -= 1
    return (y + y//4 - y//100 + y//400 + t[m-1] + d) % 7

def is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

def days_in_month(y, m):
    if m == 2:
        return 29 if is_leap(y) else 28
    if m in (4, 6, 9, 11):
        return 30
    return 31

day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

src_root = "puz"
flat_root = "nyt_puz_flat"
byweekday_root = "nyt_puz_byweekday"

os.makedirs(flat_root, exist_ok=True)
os.makedirs(byweekday_root, exist_ok=True)

for i in range(7):
    os.makedirs(os.path.join(byweekday_root, day_names[i]), exist_ok=True)

moved = 0

for year in range(1976, 2023):
    for month in range(1, 13):
        for day in range(1, days_in_month(year, month) + 1):
            src = os.path.join(src_root, str(year), "%02d" % month, "%02d.puz" % day)
            if not os.path.exists(src):
                continue

            new_name = "%04d-%02d-%02d.puz" % (year, month, day)

            dst_flat = os.path.join(flat_root, new_name)
            shutil.copy2(src, dst_flat)

            wd = weekday_sakamoto(year, month, day)
            dst_wd = os.path.join(byweekday_root, day_names[wd], new_name)
            shutil.copy2(dst_flat, dst_wd)

            moved += 1

print("Moved", moved, "files into", flat_root, "and copied into", byweekday_root)