import pyautogui
import time
"""
kaggleのデータセットを削除するコマンドがなくGUI上からのみ削除できるので、自動でGUI上から削除するコード。
pyautoguiを使ってマウスの移動とクリックをやってる.
開いてるウィンドウの位置やサイズ、ディスプレイ設定とかによるので、pyautogui.position()で逐一確認する必要がある。
"""

a = pyautogui.position()
print(a)

#Point(x=1225, y=19) datasetのタブ

# 削除したいデータセットの位置x=1475, y=471

def moveclick(x,y):
    pyautogui.moveTo(x,y)
    pyautogui.click()


for i in range(50):
    # select tab
    # x=1275 
    # y=19
    x=1569
    y=823
    moveclick(x,y)

    time.sleep(0.3)

    #select dataset
    x=1475
    y=471
    moveclick(x,y)
    # time.sleep(1)
    # ... click
    x=1838
    y=180
    moveclick(x,y)
    time.sleep(0.7)
    # Click Delete dataset
    x=1712 
    y=413
    moveclick(x,y)
    time.sleep(0.5)
    # delete
    x=1497
    y=627
    moveclick(x,y)
    time.sleep(0.5)
    # 戻るボタン
    x=601
    y=62
    moveclick(x,y)
    time.sleep(0.3)
    pyautogui.click()
    time.sleep(1.8)


# # select tab
# # x=1275 
# # y=19
# x=1569
# y=823
# moveclick(x,y)
# time.sleep(1)