# 使用 DistilBert 與 CrossAttention 做 Twitter 的情感分析
- 環境：Windows 的 wsl （Ubuntu 22.0.4）

## 步驟
1. 先在 `nlp-wsl` 底下開啟terminal（或用 cd 指令到資料夾下）
2. 在 terminal 輸入 ` python -m venv .venv` 創造建名叫 `.venv` 的虛擬環境
3. 在 terminal 輸入 `.venv/bin/pip install -r requirements.txt` 在虛擬環境安裝必要的套件
4. 在 terminal 輸入 `.venv/bin/pip install jupyterlab` 安裝執行的環境
5. 用慣用的文字編輯器到 `user資料夾/.jupyter/jupyter_notebook_config.py` 裡，找到 `#c.NotebookApp.notebook_dir = ''` 去除註解，在字串內加入路徑，修改預設路徑
6. 在 terminal 輸入 `jupyter-lab` 啟動執行環境
7. 點下全部執行按鍵，就能建立模型、進行訓練，一路執行到得到測試結果（或是直接到底下 Load Model，並執行接下來的東西）

## 注意事項
- 如果從頭開始訓練會從網路上抓 bert 的權重，會在 `user資料夾/.cache` 放置暫存檔，另外執行測試或訓練都會從網路上下載資料集，並在相同位子放置暫存檔
- 在 Windows 的環境下， `datasets` 無法正常執行，請在 Linux 或 Mac 的環境下執行本專案。
- 預設安裝的是 CPU 版的 `PyTorch` 。若有需要使用 GPU 版的，請先解除安裝後，再按照官網的教學安裝
- 使用 RTX2070 筆電版顯卡訓練需耗時約26分鐘，使用 CPU 會需耗費更多的時間