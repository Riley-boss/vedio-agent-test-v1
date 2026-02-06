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
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
MAX_RETRIES_PARSE = 4
MAX_RETRIES_DOWNLOAD = 4
MAX_DOWNLOAD_WORKERS = 4
MIN_VALID_VIDEO_BYTES = 64 * 1024

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

def create_http_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def build_parser_request_url(parser_api_url, douyin_url):
    parsed = urlparse(parser_api_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["url"] = douyin_url
    new_query = urlencode(query)
    return urlunparse(parsed._replace(query=new_query))

def _extract_video_url_from_any(data):
    if isinstance(data, str):
        url = extract_url(data)
        if url:
            return url
        return None
    if isinstance(data, list):
        for item in data:
            url = _extract_video_url_from_any(item)
            if url:
                return url
        return None
    if isinstance(data, dict):
        for key in ("url", "video_url", "play_addr", "download_url", "play_url"):
            value = data.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in data.values():
            url = _extract_video_url_from_any(value)
            if url:
                return url
    return None

def _safe_remove_dir(path):
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def download_video_via_api(douyin_url, parser_api_url):
    video_url = None
    parse_error = None
    api_full_url = build_parser_request_url(parser_api_url, douyin_url)
    session = create_http_session()

    try:
        for i in range(MAX_RETRIES_PARSE):
            try:
                headers_parse = get_random_header()
                response = session.get(api_full_url, headers=headers_parse, timeout=(8, 20))
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError:
                    data = response.text

                video_url = _extract_video_url_from_any(data)
                if video_url and video_url.startswith("http"):
                    break
                parse_error = f"未找到视频URL，响应片段: {str(data)[:200]}"
            except Exception as e:
                parse_error = str(e)

            if i < MAX_RETRIES_PARSE - 1:
                time.sleep((1.5 ** i) + random.uniform(0.2, 0.8))

        if not video_url:
            return None, f"解析彻底失败: {parse_error}"

        temp_dir = tempfile.mkdtemp(prefix="video_asr_")
        mp4_path = os.path.join(temp_dir, "video.mp4")
        download_error = None

        for i in range(MAX_RETRIES_DOWNLOAD):
            try:
                headers_download = get_random_header()
                with session.get(video_url, headers=headers_download, stream=True, timeout=(10, 60), allow_redirects=True) as video_resp:
                    video_resp.raise_for_status()
                    downloaded = 0
                    with open(mp4_path, "wb") as f:
                        for chunk in video_resp.iter_content(chunk_size=256 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)

                if downloaded < MIN_VALID_VIDEO_BYTES:
                    raise ValueError(f"下载文件过小({downloaded} bytes)，疑似无效视频")

                mp3_path = os.path.join(temp_dir, "audio.mp3")
                command = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", mp4_path,
                    "-vn",
                    "-acodec", "libmp3lame",
                    "-q:a", "4",
                    mp3_path
                ]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)
                return mp3_path, "ok"
            except subprocess.CalledProcessError as e:
                download_error = e.stderr.decode("utf-8", errors="ignore")[:200] or str(e)
            except Exception as e:
                download_error = str(e)

            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            if i < MAX_RETRIES_DOWNLOAD - 1:
                time.sleep((1.8 ** i) + random.uniform(0.2, 0.8))

        _safe_remove_dir(temp_dir)
        return None, f"下载彻底失败: {download_error}"
    finally:
        session.close()

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

def sync_to_dify(text, source_url, session):
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
        resp = session.post(url, headers=headers, json=payload, timeout=(8, 20))
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
    valid_urls = list(dict.fromkeys(valid_urls))
    
    if not valid_urls:
        st.warning("请先输入视频链接或上传文件")
    elif not SILICONFLOW_API_KEY:
        st.error("配置错误：API Key 为空，请检查 .streamlit/secrets.toml")
    else:
        client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        http_session = create_http_session()
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(valid_urls)

        max_workers = min(MAX_DOWNLOAD_WORKERS, max(1, total))
        status_text.text(f"开始并发准备音频，任务数: {total}，并发数: {max_workers}")

        completed = 0
        ordered_results = [None] * total

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(download_video_via_api, url, PARSING_API_URL): (idx, url)
                for idx, url in enumerate(valid_urls)
            }

            for future in as_completed(future_to_item):
                idx, url = future_to_item[future]
                status_text.text(f"正在转写: {url} (进度 {completed + 1}/{total})")

                try:
                    audio_path, err = future.result()
                except Exception as e:
                    audio_path, err = None, str(e)

                if audio_path and os.path.exists(audio_path):
                    transcript = transcribe_audio(client, audio_path)
                    if not transcript.startswith("转录失败"):
                        ok, msg = sync_to_dify(transcript, url, http_session)
                        sync_status = "成功" if ok else f"失败：{msg}"
                        ordered_results[idx] = {
                            "原始链接": url,
                            "状态": "成功",
                            "视频逐字稿": transcript,
                            "知识库同步状态": sync_status
                        }
                    else:
                        ordered_results[idx] = {
                            "原始链接": url,
                            "状态": "失败",
                            "视频逐字稿": "",
                            "知识库同步状态": "未同步"
                        }
                    _safe_remove_dir(os.path.dirname(audio_path))
                else:
                    st.error(f"❌ 失败: {url}\n\n**错误原因:** {err}")
                    ordered_results[idx] = {
                        "原始链接": url,
                        "状态": "失败",
                        "视频逐字稿": "",
                        "知识库同步状态": "未同步"
                    }

                completed += 1
                progress_bar.progress(completed / total)

        results = [item for item in ordered_results if item is not None]
        http_session.close()

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
