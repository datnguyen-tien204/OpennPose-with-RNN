import gdown
import os
import shutil


try:
# Face - Recogniton
#https://drive.google.com/drive/folders/1jBSMjeC2B0SaYiXQOWng6b5SBItIIp1x?usp=sharing
# Cheating - Recognition
#url2 = "https://drive.google.com/drive/folders/1o1GLOf0QMnhPg4xbcn99_x60N_KuiQnJ?usp=sharing"
    url = "https://drive.google.com/drive/folders/1jBSMjeC2B0SaYiXQOWng6b5SBItIIp1x?usp=sharing"
    gdown.download_folder(url, quiet=False, use_cookies=False)

    # Cheating- Recogintion
    url2 = "https://drive.google.com/drive/folders/1o1GLOf0QMnhPg4xbcn99_x60N_KuiQnJ?usp=sharing"
    gdown.download_folder(url2, quiet=False, use_cookies=False)

    print("Download requirements file successfully")

except Exception as e:
    print("Download requirements file failed")
    print(e)


print(f"All files requirement setup successfully")
