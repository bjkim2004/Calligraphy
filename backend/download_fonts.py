"""
Korean Handwriting Fonts Download Script
Using Google Fonts API
"""

import os
import sys
import urllib.request
import ssl

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ssl._create_default_https_context = ssl._create_unverified_context

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

FONT_URLS = {
    "EastSeaDokdo.ttf": "https://fonts.gstatic.com/s/eastseadokdo/v22/xfuo0Wn2V2_KanASqXSZp22m05_aGavYS18y.ttf",
    "NanumBrushScript.ttf": "https://fonts.gstatic.com/s/nanumbrushscript/v22/wXK2E2wfpokopxzthSqPbcR5_gVaxazyjqBr1lO97Q.ttf",
    "SongMyung.ttf": "https://fonts.gstatic.com/s/songmyung/v20/1cX2aUDWAJH5-EIC7DIhr1GqhcitzeM.ttf",
    "GowunBatang.ttf": "https://fonts.gstatic.com/s/gowunbatang/v7/ijwSs5nhRMIjYsdSgcMa3wRhXLH-yuAtLw.ttf",
    "GowunBatangBold.ttf": "https://fonts.gstatic.com/s/gowunbatang/v7/ijwNs5nhRMIjYsdSgcMa3wRZ4J7awssxJii23w.ttf",
    "Hahmlet.ttf": "https://fonts.gstatic.com/s/hahmlet/v13/BngXUXpCQ3nKpIo0TfPyfCdXfaeU4RhKOdjobsO-aVxn.ttf",
    "Gaegu.ttf": "https://fonts.gstatic.com/s/gaegu/v17/TuGSUVB6Up9NU57nifw74sdtBk0x.ttf",
    "HiMelody.ttf": "https://fonts.gstatic.com/s/himelody/v15/46ktlbP8Vnz0pJcqCTbEf29E31BBGA.ttf",
    "SingleDay.ttf": "https://fonts.gstatic.com/s/singleday/v17/LYjHdGDjlEgoAcF95EI5jVoFUNfeQJU.ttf",
    "NanumPenScript.ttf": "https://fonts.gstatic.com/s/nanumpenscript/v19/daaDSSYiLGqEal3MvdA_FOL_3FkN2z7-aMFCcTU.ttf",
    "CuteFont.ttf": "https://fonts.gstatic.com/s/cutefont/v22/Noaw6Uny2oWPbSHMrY6vmJNVNC9hkw.ttf",
    "GamjaFlower.ttf": "https://fonts.gstatic.com/s/gamjaflower/v22/6NUR8FiKJg-Pa0rM6uN40Z4kyf9Fdty2ew.ttf",
    "Jua.ttf": "https://fonts.gstatic.com/s/jua/v17/co3KmW9ljjAjc-DZCsKgsg.ttf",
    "DoHyeon.ttf": "https://fonts.gstatic.com/s/dohyeon/v19/TwMN-I8CRRU2zM86HFE3ZwaH__-C.ttf",
    "GothicA1.ttf": "https://fonts.gstatic.com/s/gothica1/v13/CSR44z5ZnPydRjlCCwlCmOQKSPl6tOU9Eg.ttf",
    "Stylish.ttf": "https://fonts.gstatic.com/s/stylish/v22/m8JSjfhPYriQkk7-fo35dLxEdmo.ttf",
    "NotoSansKR.ttf": "https://fonts.gstatic.com/s/notosanskr/v36/PbyxFmXiEBPT4ITbgNA5Cgms3VYcOA-vvnIzzuoyeLTq8H4hfeE.ttf",
    "BlackHanSans.ttf": "https://fonts.gstatic.com/s/blackhansans/v17/ea8Aad44WunzF9a-dL6toA8r8nqVIXSkH-Hc.ttf",
}


def download_fonts():
    os.makedirs(FONTS_DIR, exist_ok=True)
    
    print("[Download] Downloading Korean handwriting fonts...")
    print(f"[Info] Total {len(FONT_URLS)} fonts\n")
    
    success_count = 0
    fail_count = 0
    
    for font_name, url in FONT_URLS.items():
        font_path = os.path.join(FONTS_DIR, font_name)
        
        if os.path.exists(font_path):
            size = os.path.getsize(font_path)
            if size > 10000:
                print(f"  [OK] {font_name} (already exists)")
                success_count += 1
                continue
        
        try:
            print(f"  [Downloading] {font_name}...")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(font_path, 'wb') as f:
                    f.write(response.read())
            
            size = os.path.getsize(font_path)
            if size > 10000:
                print(f"  [OK] {font_name} ({size:,} bytes)")
                success_count += 1
            else:
                os.remove(font_path)
                print(f"  [Error] {font_name} - file too small")
                fail_count += 1
        except Exception as e:
            print(f"  [Error] {font_name} - {str(e)[:60]}")
            fail_count += 1
    
    print(f"\n[Done] Download complete!")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {fail_count}")
    print(f"[Path] {FONTS_DIR}")


if __name__ == "__main__":
    download_fonts()
