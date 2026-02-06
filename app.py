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
MAX_PARSE_CANDIDATES = 8

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
        "Referer": "https://www.douyin.com/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive"
    }

def extract_url(text):
    if not text:
        return None
    pattern = r'(https?://[^\s]+)'
    match = re.search(pattern, text)
    return match.group(1) if match else None

def sanitize_url(raw_url):
    if not raw_url:
        return ""
    cleaned = str(raw_url).strip()
    cleaned = cleaned.rstrip("，。；、!?！？\"'】）)>")
    return cleaned

def extract_video_id(url):
    if not url:
        return None
    patterns = [
        r"/video/(\d+)",
        r"/share/video/(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    try:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        for key in ("item_id", "aweme_id"):
            if query.get(key, "").isdigit():
                return query[key]
    except Exception:
        pass
    return None

def resolve_redirect_url(session, url):
    try:
        headers = get_random_header()
        resp = session.get(url, headers=headers, timeout=(8, 20), allow_redirects=True)
        final_url = sanitize_url(resp.url)
        return final_url or sanitize_url(url)
    except Exception:
        return sanitize_url(url)

def build_parse_candidates(session, douyin_url):
    candidates = []
    seen = set()

    def _add(candidate):
        value = sanitize_url(candidate)
        if value and value not in seen:
            candidates.append(value)
            seen.add(value)

    base_url = sanitize_url(douyin_url)
    _add(base_url)

    redirected_url = resolve_redirect_url(session, base_url)
    _add(redirected_url)

    video_id = extract_video_id(base_url) or extract_video_id(redirected_url)
    if video_id:
        _add(f"https://www.douyin.com/video/{video_id}")
        _add(f"https://www.douyin.com/share/video/{video_id}/")
        _add(f"https://www.iesdouyin.com/share/video/{video_id}/")
        _add(f"https://www.douyin.com/discover?modal_id={video_id}")

    return candidates[:MAX_PARSE_CANDIDATES]

def normalize_cell_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def extract_url_from_row(row_data, preferred_col=""):
    if preferred_col and preferred_col in row_data:
        url = extract_url(row_data.get(preferred_col, ""))
        if url:
            return url

    for candidate in ("视频链接", "链接", "URL", "url", "video_url", "作品链接", "抖音链接"):
        if candidate in row_data:
            url = extract_url(row_data.get(candidate, ""))
            if url:
                return url

    for value in row_data.values():
        url = extract_url(value)
        if url:
            return url
    return None

def build_dify_document_text(text, source_url, metadata):
    lines = [build_structured_summary(metadata, source_url)]
    lines.extend([
        "",
        "【元数据】",
        f"视频来源: {source_url}",
        f"处理时间: {time.strftime('%Y-%m-%d')}",
    ])
    for key, value in metadata.items():
        if value:
            lines.append(f"{key}: {value}")
    lines.extend([
        "",
        "【转写正文开始】",
        text if text is not None else "",
        "【转写正文结束】",
    ])
    return "\n".join(lines)

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

def is_valid_video_content_type(content_type):
    if not content_type:
        return False
    value = content_type.lower()
    return ("video/" in value) or ("octet-stream" in value)

def transcript_effective_length(text):
    if not text:
        return 0
    return len(re.sub(r"\s+", "", str(text)))

def remove_emoji(text):
    if not text:
        return ""
    emoji_pattern = re.compile(
        "[" 
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_pattern.sub("", str(text))
    cleaned = re.sub(r"[\u200d\ufe0f]", "", cleaned)
    return cleaned.strip()

def normalize_numeric_value(raw_value):
    if raw_value is None:
        return ""
    text = str(raw_value).strip().replace(",", "")
    if text == "":
        return ""
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return ""
    return match.group(0)

def extract_video_id_from_url(url):
    return extract_video_id(url) or ""

def build_document_name(metadata, source_url):
    dealer_id = metadata.get("经销商ID", "").strip() or "unknownDealer"
    title = metadata.get("视频标题", "").strip() or "untitled"
    video_id = extract_video_id_from_url(source_url) or f"ts{int(time.time())}"
    safe_title = re.sub(r"[\\/:*?\"<>|]", "_", title)[:30]
    return f"dealer_{dealer_id}__video_{video_id}__{safe_title}"

def build_structured_summary(metadata, source_url):
    fields = {
        "video_id": extract_video_id_from_url(source_url),
        "dealer_id": metadata.get("经销商ID", ""),
        "dealer_name": metadata.get("经销商简称", ""),
        "platform": metadata.get("社媒平台", ""),
        "publish_time": metadata.get("发布时间", ""),
        "title": metadata.get("视频标题", ""),
        "car_series": metadata.get("视频车系", ""),
        "play_count": normalize_numeric_value(metadata.get("播放量", "")),
        "like_count": normalize_numeric_value(metadata.get("点赞量", "")),
        "comment_count": normalize_numeric_value(metadata.get("评论量", "")),
        "share_count": normalize_numeric_value(metadata.get("分享量", "")),
        "engagement_count": normalize_numeric_value(metadata.get("互动量", "")),
    }
    lines = ["【结构化字段】"]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)

def download_video_via_api(douyin_url, parser_api_url):
    video_url = None
    parse_error = None
    session = create_http_session()

    try:
        parse_candidates = build_parse_candidates(session, douyin_url)
        for candidate_idx, candidate_url in enumerate(parse_candidates):
            api_full_url = build_parser_request_url(parser_api_url, candidate_url)
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
                    parse_error = f"候选{candidate_idx + 1}解析失败，响应片段: {str(data)[:200]}"
                except Exception as e:
                    parse_error = str(e)

                if i < MAX_RETRIES_PARSE - 1:
                    time.sleep((1.4 ** i) + random.uniform(0.2, 0.7))

            if video_url:
                break
            if candidate_idx < len(parse_candidates) - 1:
                time.sleep(random.uniform(0.2, 0.6))

        if not video_url:
            return None, f"解析彻底失败: {parse_error}，已尝试候选链接数: {len(parse_candidates)}"

        temp_dir = tempfile.mkdtemp(prefix="video_asr_")
        mp4_path = os.path.join(temp_dir, "video.mp4")
        download_error = None

        for i in range(MAX_RETRIES_DOWNLOAD):
            try:
                headers_download = get_random_header()
                with session.get(video_url, headers=headers_download, stream=True, timeout=(10, 60), allow_redirects=True) as video_resp:
                    video_resp.raise_for_status()
                    content_type = video_resp.headers.get("Content-Type", "")
                    if not is_valid_video_content_type(content_type):
                        raise ValueError(f"下载内容类型异常: {content_type}")
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

def sync_to_dify(text, source_url, metadata, session):
    try:
        api_key = DIFY_API_KEY if 'DIFY_API_KEY' in globals() else ""
        dataset_id = DIFY_DATASET_ID if 'DIFY_DATASET_ID' in globals() else ""
        if not api_key or not dataset_id:
            return False, "Dify配置缺失", ""
        url = f"https://api.dify.ai/v1/datasets/{dataset_id}/document/create_by_text"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": build_document_name(metadata, source_url),
            "text": build_dify_document_text(text, source_url, metadata),
            "indexing_technique": "high_quality",
            "process_rule": {"mode": "automatic"}
        }
        resp = session.post(url, headers=headers, json=payload, timeout=(8, 20))
        if 200 <= resp.status_code < 300:
            doc_id = ""
            try:
                data = resp.json()
                if isinstance(data, dict):
                    doc_id = str(
                        data.get("id")
                        or data.get("document_id")
                        or data.get("data", {}).get("id")
                        or data.get("data", {}).get("document", {}).get("id")
                        or ""
                    )
            except Exception:
                pass
            return True, "同步成功", doc_id
        else:
            err_msg = ""
            try:
                data = resp.json()
                err_msg = data.get("message") or str(data)
            except Exception:
                err_msg = resp.text
            return False, err_msg[:200], ""
    except Exception as e:
        return False, str(e), ""

# --- UI 侧边栏（仅保留输入区域） ---
uploaded_file = st.sidebar.file_uploader("Excel/CSV 批量上传", type=["xlsx", "xls", "csv"])
input_text = st.sidebar.text_area("或输入链接 (一行一个)", height=100)
preferred_url_column = st.sidebar.text_input("链接列名（可选）", value="链接")
dealer_id_column = st.sidebar.text_input("经销商ID列名（可选）", value="经销商ID")
enable_dedup = st.sidebar.checkbox("按经销商ID+链接去重（保序）", value=True)

# --- UI 主界面 ---
st.title("批量视频转文字工具")

if st.button("开始处理", type="primary"):
    input_records = []

    if input_text.strip():
        lines = input_text.strip().split("\n")
        for line_no, line in enumerate(lines, start=1):
            url = extract_url(line)
            if url:
                input_records.append({
                    "video_url": url,
                    "fields": {
                        "输入来源": "手动输入",
                        "源行号": str(line_no),
                        "原始输入": line.strip()
                    }
                })

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            selected_url_col = preferred_url_column.strip()
            for row_idx, row in df_upload.iterrows():
                row_data = {str(col): normalize_cell_value(row[col]) for col in df_upload.columns}
                url = extract_url_from_row(row_data, preferred_col=selected_url_col)
                if url:
                    row_data["输入来源"] = "文件上传"
                    row_data["源文件名"] = uploaded_file.name
                    row_data["源行号"] = str(row_idx + 2)
                    input_records.append({
                        "video_url": url,
                        "fields": row_data
                    })
        except Exception as e:
            st.error(f"读取文件失败: {e}")

    process_records = input_records
    if enable_dedup and process_records:
        deduped_records = []
        seen = set()
        selected_dealer_col = dealer_id_column.strip()

        for record in process_records:
            dealer_value = ""
            if selected_dealer_col:
                dealer_value = record["fields"].get(selected_dealer_col, "")
            dedupe_key = (dealer_value, record["video_url"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped_records.append(record)

        if len(deduped_records) < len(process_records):
            st.info(f"已按经销商ID+链接去重：{len(process_records)} -> {len(deduped_records)}")
        process_records = deduped_records

    if not process_records:
        st.warning("请先输入视频链接或上传文件")
    elif not SILICONFLOW_API_KEY:
        st.error("配置错误：API Key 为空，请检查 .streamlit/secrets.toml")
    else:
        client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        http_session = create_http_session()
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(process_records)

        max_workers = min(MAX_DOWNLOAD_WORKERS, max(1, total))
        status_text.text(f"开始并发准备音频，任务数: {total}，并发数: {max_workers}")

        completed = 0
        ordered_results = [None] * total

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(download_video_via_api, record["video_url"], PARSING_API_URL): (idx, record)
                    for idx, record in enumerate(process_records)
                }

                for future in as_completed(future_to_item):
                    idx, record = future_to_item[future]
                    url = record["video_url"]
                    result_row = dict(record["fields"])
                    result_row["原始链接"] = url
                    status_text.text(f"正在转写: {url} (进度 {completed + 1}/{total})")

                    try:
                        audio_path, err = future.result()
                    except Exception as e:
                        audio_path, err = None, str(e)

                    if audio_path and os.path.exists(audio_path):
                        transcript = transcribe_audio(client, audio_path)
                        if not transcript.startswith("转录失败"):
                            cleaned_transcript = remove_emoji(transcript)
                            effective_len = transcript_effective_length(cleaned_transcript)
                            result_row["转写状态"] = "成功"
                            result_row["视频逐字稿"] = cleaned_transcript
                            result_row["转写字符数"] = effective_len
                            metadata_for_dify = {
                                key: value for key, value in result_row.items()
                                if value and key not in {
                                    "视频逐字稿", "转写状态", "知识库同步状态", "错误原因", "Dify文档ID",
                                    "转写字符数", "是否入库"
                                }
                            }
                            ok, msg, doc_id = sync_to_dify(cleaned_transcript, url, metadata_for_dify, http_session)
                            result_row["是否入库"] = "是" if ok else "否"
                            result_row["知识库同步状态"] = "成功" if ok else f"失败：{msg}"
                            result_row["Dify文档ID"] = doc_id
                            result_row["错误原因"] = "" if ok else msg
                        else:
                            result_row["转写状态"] = "失败"
                            result_row["视频逐字稿"] = ""
                            result_row["转写字符数"] = 0
                            result_row["是否入库"] = "否"
                            result_row["知识库同步状态"] = "未同步"
                            result_row["Dify文档ID"] = ""
                            result_row["错误原因"] = transcript
                        _safe_remove_dir(os.path.dirname(audio_path))
                    else:
                        st.error(f"❌ 失败: {url}\n\n**错误原因:** {err}")
                        result_row["转写状态"] = "失败"
                        result_row["视频逐字稿"] = ""
                        result_row["转写字符数"] = 0
                        result_row["是否入库"] = "否"
                        result_row["知识库同步状态"] = "未同步"
                        result_row["Dify文档ID"] = ""
                        result_row["错误原因"] = err

                    ordered_results[idx] = result_row
                    completed += 1
                    progress_bar.progress(completed / total)
        finally:
            http_session.close()

        results = [item for item in ordered_results if item is not None]
        hidden_output_fields = {"输入来源", "源文件名"}
        for row in results:
            for field in hidden_output_fields:
                row.pop(field, None)

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
