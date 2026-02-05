import streamlit as st
import requests
import os
import re
import pandas as pd
from openai import OpenAI
import tempfile
import subprocess
from io import BytesIO
import time
import random
import json

# --- 核心配置区 (安全读取) ---
try:
    # 尝试读取
    SILICONFLOW_API_KEY = st.secrets["SILICONFLOW_API_KEY"]
    PARSING_API_URL = st.secrets["PARSING_API_URL"]
    DIFY_API_KEY = st.secrets.get("DIFY_API_KEY", "")
    DIFY_DATASET_ID = st.secrets.get("DIFY_DATASET_ID", "")
    print(f"DEBUG: 解析API地址: {PARSING_API_URL}, SiliconFlowKey长度: {len(SILICONFLOW_API_KEY) if SILICONFLOW_API_KEY else 0}")

except Exception as e:
    st.error(f"❌ 启动失败: {e}")
    st.error("请检查 .streamlit/secrets.toml (本地) 或 Streamlit Cloud Secrets (云端)。")
    st.stop()
# ---------------------------------------

st.set_page_config(page_title="批量视频转文字工具", layout="wide")

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
ASR_MODEL = "iic/SenseVoiceSmall"

# UA 池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]

def get_random_header():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://www.douyin.com/"
    }

def extract_url(text):
    if not text:
        return None
    pattern = r'(https?://[^\s]+)'
    match = re.search(pattern, text)
    return match.group(1) if match else None

def download_video_via_api(douyin_url, parser_api_url, status_callback=None):
    # 重试参数
    MAX_RETRIES_PARSE = 3
    MAX_RETRIES_DOWNLOAD = 3
    
    # 1. 解析阶段重试
    video_url = None
    parse_error = None
    
    if '?' in parser_api_url:
        api_full_url = f"{parser_api_url}&url={douyin_url}"
    else:
        api_full_url = f"{parser_api_url}?url={douyin_url}"
        
    for i in range(MAX_RETRIES_PARSE):
        try:
            if i > 0 and status_callback:
                status_callback(f":gray[连接不稳定，正在重试... ({i+1}/{MAX_RETRIES_PARSE})]")
            
            headers_parse = get_random_header()
            print(f"DEBUG: 请求解析API URL: {api_full_url}")
            response = requests.get(api_full_url, headers=headers_parse, timeout=15)
            print(f"DEBUG: 解析API返回状态: {response.status_code}, 响应前200字符: {response.text[:200]}")
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict):
                    video_url = data["data"].get("url") or data["data"].get("play_addr")
                elif "url" in data:
                    video_url = data["url"]
                elif "video_url" in data:
                    video_url = data["video_url"]
            
            if video_url:
                break # 成功拿到 URL
            else:
                parse_error = f"未找到视频 URL. 返回: {str(data)[:200]}"
                time.sleep(3) # 解析失败等待
                
        except Exception as e:
            parse_error = str(e)
            time.sleep(3) # 异常等待
            
    if not video_url:
        return None, f"解析彻底失败: {parse_error}"

    # 2. 下载阶段重试
    temp_dir = tempfile.mkdtemp()
    mp4_path = os.path.join(temp_dir, "video.mp4")
    download_success = False
    download_error = None
    
    for i in range(MAX_RETRIES_DOWNLOAD):
        try:
            if i > 0 and status_callback:
                status_callback(f":gray[连接不稳定，正在重试... ({i+1}/{MAX_RETRIES_DOWNLOAD})]")
                
            headers_download = get_random_header()
            video_resp = requests.get(video_url, headers=headers_download, stream=True, timeout=30)
            video_resp.raise_for_status()
            
            with open(mp4_path, "wb") as f:
                for chunk in video_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            download_success = True
            break
        except Exception as e:
            download_error = str(e)
            time.sleep(5) # 下载失败等待
            
    if not download_success:
        return None, f"下载彻底失败: {download_error}"

    # 3. 转码音频
    try:
        mp3_path = os.path.join(temp_dir, "audio.mp3")
        command = [
            "ffmpeg", "-y",
            "-i", mp4_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "4",
            mp3_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(mp4_path) # 立即删除 mp4
        return mp3_path, "ok"
    except Exception as e:
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
        return None, f"转码失败: {str(e)}"

def transcribe_audio(client, file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        return f"转录失败: {str(e)}"

def sync_to_dify(text, source_url):
    try:
        api_key = DIFY_API_KEY if 'DIFY_API_KEY' in globals() else ""
        dataset_id = DIFY_DATASET_ID if 'DIFY_DATASET_ID' in globals() else ""
        if not api_key or not dataset_id:
            return False, "Dify配置缺失"
        url = f"https://api.dify.ai/v1/datasets/{dataset_id}/document/create_by_text"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": f"视频分析_{int(time.time())}",
            "text": f"【视频来源】：{source_url}\n【处理时间】：{time.strftime('%Y-%m-%d')}\n\n{text}",
            "indexing_technique": "high_quality",
            "process_rule": {"mode": "automatic"}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if 200 <= resp.status_code < 300:
            return True, "同步成功"
        else:
            err_msg = ""
            try:
                data = resp.json()
                err_msg = data.get("message") or str(data)
            except Exception:
                err_msg = resp.text
            return False, err_msg[:200]
    except Exception as e:
        return False, str(e)

# --- UI 侧边栏（仅保留输入区域） ---
uploaded_file = st.sidebar.file_uploader("Excel/CSV 批量上传", type=["xlsx", "xls", "csv"])
input_text = st.sidebar.text_area("或输入链接 (一行一个)", height=100)

# --- UI 主界面 ---
st.title("批量视频转文字工具")

if st.button("开始处理", type="primary"):
    valid_urls = []
    if input_text.strip():
        lines = input_text.strip().split("\n")
        for line in lines:
            url = extract_url(line)
            if url:
                valid_urls.append(url)
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            for _, row in df_upload.iterrows():
                row_str = " ".join(row.astype(str).values)
                url = extract_url(row_str)
                if url:
                    valid_urls.append(url)
        except Exception as e:
            st.error(f"读取文件失败: {e}")
    valid_urls = list(set(valid_urls))
    
    if not valid_urls:
        st.warning("请先输入视频链接或上传文件")
    elif not SILICONFLOW_API_KEY:
        st.error("配置错误：API Key 为空，请检查 .streamlit/secrets.toml")
    else:
        client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(valid_urls)
        
        for i, url in enumerate(valid_urls):
            current_progress_text = f"正在处理: {url} (进度 {i+1}/{total})"
            status_text.text(current_progress_text)
            
            # 定义回调函数更新 UI 状态
            def update_status(msg):
                status_text.markdown(msg) # 使用 markdown 支持颜色
                
            audio_path, err = download_video_via_api(url, PARSING_API_URL, status_callback=update_status)
            
            if audio_path and os.path.exists(audio_path):
                # 状态保持显示正在处理，不刷屏
                transcript = transcribe_audio(client, audio_path)
                if not transcript.startswith("转录失败"):
                    ok, msg = sync_to_dify(transcript, url)
                    if ok:
                        update_status(":green[Dify同步成功]")
                        sync_status = "成功"
                    else:
                        update_status(f":red[Dify同步失败：{msg}]")
                        sync_status = f"失败：{msg}"
                    results.append({
                        "原始链接": url,
                        "状态": "成功",
                        "视频逐字稿": transcript,
                        "知识库同步状态": sync_status
                    })
                else:
                    results.append({
                        "原始链接": url,
                        "状态": "失败",
                        "视频逐字稿": "",
                        "知识库同步状态": "未同步"
                    })
                try:
                    os.remove(audio_path)
                    os.rmdir(os.path.dirname(audio_path))
                except:
                    pass
            else:
                st.error(f"❌ 失败: {url}\n\n**错误原因:** {err}")
                results.append({
                    "原始链接": url,
                    "状态": "失败",
                    "视频逐字稿": "",
                    "知识库同步状态": "未同步"
                })
            
            progress_bar.progress((i + 1) / total)
            
            # 防风控：随机延时 (静默处理，不显示倒计时干扰用户)
            if i < total - 1:
                sleep_time = random.uniform(2, 5)
                time.sleep(sleep_time)

        status_text.text("处理完成")
        
        if results:
            df = pd.DataFrame(results)
            
            # 导出按钮置于表格上方
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="transcripts")
                
            st.download_button(
                label="导出结果 (Excel)",
                data=buffer.getvalue(),
                file_name="transcripts.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
            st.dataframe(df, hide_index=True)
