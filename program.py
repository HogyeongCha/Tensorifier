import streamlit as st
import pandas as pd
import numpy as np
import torch
import google.generativeai as genai
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import os


# --- [1. 환경 설정 및 API 연결] ---
st.set_page_config(page_title="Data Preprocessing Agent", layout="centered")

GEMINI_API_KEY = None

# .streamlit/secrets.toml 파일이 존재하는지 먼저 확인 (에러 박스 방지)
secrets_exist = False
try:
    if os.path.exists(".streamlit/secrets.toml"):
        secrets_exist = True
    elif os.path.exists(os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")):
        secrets_exist = True
except:
    pass

if secrets_exist:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except:
        pass

if not GEMINI_API_KEY:
    # secrets가 없으면 사용자에게 입력을 받거나 에러를 띄웁니다 (보안상 안전)
    GEMINI_API_KEY = st.text_input("Gemini API Key를 입력하세요:", type="password")
    if not GEMINI_API_KEY:
        st.warning("API Key가 설정되지 않았습니다. 로컬 실행 시 .streamlit/secrets.toml 파일을 확인하세요.")
        st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')



# Toss 스타일 커스텀 CSS (단계별 진행 바 등 추가)
st.markdown("""
    <style>
    .main { background-color: #F9FAFB; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3em; background-color: #0047FF; color: white; font-weight: bold; border: none; }
    .stProgress > div > div > div > div { background-color: #0047FF; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; border: 1px solid #E5E7EB; }
    </style>
""", unsafe_allow_html=True)

# --- [2. 도구 함수들] ---
def get_agent_plan(df, goal):
    """Gemini가 전체 전처리 계획을 수립합니다."""
    sample_data = df.head(5).to_csv(index=False)
    null_info = df.isnull().mean().to_dict()
    column_types = df.dtypes.astype(str).to_dict()
    
    prompt = f"""
    당신은 데이터 전처리 전문가 에이전트입니다.
    사용자의 목표: {goal}
    데이터 샘플: {sample_data}
    결측률: {null_info}
    타입: {column_types}
    
    각 컬럼별로 [Drop, Fill_Median, Fill_Mode, Fill_Zero, Normalize, Pass] 중 하나를 선택해 단계별 계획을 세우세요.
    반드시 JSON 리스트 형식으로만 응답하세요.
    [
        {{"col": "컬럼명", "action": "선택한 액션", "reason": "이유(한글)"}}
    ]
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error calling Gemini: {e}") # 터미널 로그 출력
        # Fallback (비상용 기본 계획)
        return [{"col": c, "action": "Pass", "reason": f"API 오류로 기본 Pass 처리 ({str(e)})"} for c in df.columns[:3]]

def apply_step(df, col, action):
    """단일 스텝(컬럼 액션)을 수행하고 결과 데이터프레임을 반환합니다."""
    new_df = df.copy()
    
    if action == "Drop":
        new_df = new_df.drop(columns=[col])
    elif action == "Fill_Median":
        if pd.api.types.is_numeric_dtype(new_df[col]):
            val = new_df[col].median()
            new_df[col] = new_df[col].fillna(val)
    elif action == "Fill_Mode":
        val = new_df[col].mode()[0]
        new_df[col] = new_df[col].fillna(val)
    elif action == "Fill_Zero":
        new_df[col] = new_df[col].fillna(0)
    elif action == "Normalize":
        if pd.api.types.is_numeric_dtype(new_df[col]):
            scaler = StandardScaler()
            # 2D reshape 필요
            data = new_df[[col]].values
            new_df[col] = scaler.fit_transform(data).flatten()
            
    return new_df

def plot_comparison(old_df, new_df, col):
    """변경 전후 분포 비교 시각화 (Enhanced)"""
    if col not in old_df.columns or col not in new_df.columns:
        return

    # 1. 수치형 데이터 시각화
    if pd.api.types.is_numeric_dtype(new_df[col]):
        st.markdown(f"##### 📊 {col} 수치형 분포 비교")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # KDE Plot
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.kdeplot(old_df[col].dropna(), color="gray", linestyle="--", label="Original", ax=ax1)
            sns.kdeplot(new_df[col], color="blue", fill=True, alpha=0.3, label="Transformed", ax=ax1)
            ax1.set_title("Density Distribution")
            ax1.set_xlabel("Value")
            ax1.legend()
            st.pyplot(fig1)
            
        with col_c2:
            # Box Plot
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            # Boxplot needs list of arrays, handling NaNs for original
            data_org = old_df[col].dropna().values
            data_new = new_df[col].values # Transformed shouldn't have NaNs usually, or handled
            ax2.boxplot([data_org, data_new], labels=['Original', 'Transformed'])
            ax2.set_title("Box Plot (Outliers)")
            st.pyplot(fig2)
            
        # 통계 요약 (가로로 배치)
        # 통계 요약 (가로로 배치)
        st.markdown("#### 🔢 상세 통계 변화")
        desc_old = old_df[col].describe()
        desc_new = new_df[col].describe()
        stats_df = pd.DataFrame({'Original': desc_old, 'Transformed': desc_new})
        st.dataframe(stats_df.T, use_container_width=True)

    # 2. 범주형 데이터 시각화
    else:
        st.markdown(f"##### 📊 {col} 범주형 빈도 비교")
        
        # 상위 10개 카테고리만 비교
        top_n = 10
        top_cats = old_df[col].value_counts().head(top_n).index
        if len(top_cats) == 0:
             top_cats = new_df[col].value_counts().head(top_n).index
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 빈도 계산
        old_counts = old_df[col].value_counts()
        new_counts = new_df[col].value_counts()
        
        # DataFrame으로 병합
        cat_df = pd.DataFrame({'Original': old_counts, 'Transformed': new_counts})
        # 상위 카테고리만 필터링
        cat_df = cat_df.loc[cat_df.index.intersection(top_cats)].fillna(0)
        
        cat_df.plot(kind='bar', ax=ax, color=['gray', 'blue'], alpha=0.7, rot=45)
        ax.set_title(f"Top {top_n} Category Frequencies")
        st.pyplot(fig)

# --- [3. 메인 로직] ---
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
    st.session_state.current_step_idx = 0
    st.session_state.log = [] # 수행 로그

st.title("🤖 Data Preprocessing A.I. Agent")

# [Step 1] 데이터 및 목표 설정
if st.session_state.step == 'upload':
    st.subheader("1. 데이터 분석 시작하기")
    uploaded_file = st.file_uploader("분석할 데이터를 주세요.", type="csv")
    user_goal = st.text_input("목표를 알려주세요.", placeholder="예: 생존자 예측")
    
    if uploaded_file and user_goal:
        if st.button("계획 수립 요청"):
            st.session_state.original_df = pd.read_csv(uploaded_file)
            st.session_state.df = st.session_state.original_df.copy() # 작업용
            st.session_state.history = [] # 되돌리기 위한 히스토리 스택 (df 상태 저장)
            st.session_state.goal = user_goal
            
            with st.spinner("AI가 데이터를 훑어보고 최적의 계획을 짜는 중..."):
                plan = get_agent_plan(st.session_state.df, user_goal)
                st.session_state.plan = plan
                st.session_state.step = 'execute_loop'
                st.rerun()

# [Step 2] 단계별 실행 및 검증 (핵심 로직 변경)
elif st.session_state.step == 'execute_loop':
    plan = st.session_state.plan
    idx = st.session_state.current_step_idx
    
    # 계획이 비어있는 경우 예외 처리
    if not plan:
        st.error("Gemini가 계획을 수립하지 못했습니다. (응답 오류 또는 빈 데이터)")
        if st.button("다시 시도하기"):
            st.session_state.step = 'upload'
            st.rerun()
        st.stop()

    # 진행률 표시
    progress = (idx / len(plan))
    st.progress(progress, text=f"전체 계획 진행률: {int(progress*100)}%")
    
    if idx < len(plan):
        current_item = plan[idx]
        col = current_item['col']
        action = current_item['action']
        reason = current_item['reason']
        
        st.subheader(f"Step {idx+1}. {col} 처리")
        
        # 1. 에이전트의 제안 설명
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"**'{col}'** 컬럼에 **'{action}'** 작업을 수행하겠습니다.")
            st.caption(f"이유: {reason}")
        
        # 2. 결과 미리보기 (Preview)
        # 현재 확정된 df를 기준으로 이번 스텝만 적용해봄
        preview_df = apply_step(st.session_state.df, col, action)
        
        with st.expander("🔍 수행 결과 미리보기 (Before vs After)", expanded=True):
            # 수치형인 경우 그래프 비교
            if action != "Drop": # Drop이면 비교 불가하거나 사라짐 표시
                plot_comparison(st.session_state.df, preview_df, col)
            else:
                st.warning(f"'{col}' 컬럼이 데이터에서 완전히 제거됩니다.")
                
            col1, col2 = st.columns(2)
            col1.metric("변경 전 결측치", st.session_state.df[col].isnull().sum() if col in st.session_state.df else 0)
            col2.metric("변경 후 결측치", preview_df[col].isnull().sum() if col in preview_df else 0)

        # 3. 사용자 승인 인터페이스
        st.write("---")
        col_accept, col_reject, col_undo = st.columns([1, 1, 1])
        with col_accept:
            if st.button(f"✅ 승인 ({action})", key="btn_accept", use_container_width=True):
                # 변경 전 상태 저장
                st.session_state.history.append(st.session_state.df.copy())
                
                # 확정(Commit)
                st.session_state.df = preview_df
                st.session_state.log.append(f"Step {idx+1}: {col} -> {action} 완료")
                st.session_state.current_step_idx += 1
                st.rerun()
        
        with col_reject:
            if st.button("❌ 거절 (변경 안함)", key="btn_reject", use_container_width=True):
                # 변경 전 상태 저장 (Pass의 경우도 상태 저장은 필요, idx를 되돌려야 하므로)
                st.session_state.history.append(st.session_state.df.copy())
                
                # 변경 없이 다음 단계로
                st.session_state.log.append(f"Step {idx+1}: {col} -> Pass (사용자 거절)")
                st.session_state.current_step_idx += 1
                st.rerun()

        with col_undo:
            if idx > 0:
                if st.button("↩️ 되돌리기", key="btn_undo", use_container_width=True):
                    # 이전 상태 복구
                    st.session_state.df = st.session_state.history.pop()
                    st.session_state.log.pop()
                    st.session_state.current_step_idx -= 1
                    st.rerun()
            
    else:
        # 모든 계획 수행 완료
        st.session_state.step = 'final'
        st.rerun()

# [Step 3] 최종 완료 및 텐서 변환
elif st.session_state.step == 'final':
    st.balloons()
    st.subheader("🎉 모든 전처리 단계가 완료되었습니다!")
    
    with st.expander("📜 수행된 작업 로그 확인"):
        for log_item in st.session_state.log:
            st.write(f"- {log_item}")
            
    if st.button("최종 텐서(Tensor) 생성"):
        df = st.session_state.df
        # 수치형만 남기기 (텐서 변환용)
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_df.empty:
            st.error("남은 수치형 데이터가 없습니다.")
        else:
            # 학습/테스트 분리
            X_train, X_test = train_test_split(numeric_df, test_size=0.2, random_state=42)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            
            st.success(f"생성 완료! Device: {device}")
            st.code(f"Train Tensor Shape: {X_train_tensor.shape}\nTest Tensor Shape: {X_test_tensor.shape}")

            # 텐서 파일 저장 (.pt)
            tensor_buffer = io.BytesIO()
            torch.save({
                'X_train': X_train_tensor,
                'X_test': X_test_tensor,
                'columns': list(numeric_df.columns)
            }, tensor_buffer)
            
            st.download_button(
                label="💾 텐서 파일 다운로드 (tensors.pt)",
                data=tensor_buffer.getvalue(),
                file_name="tensors.pt",
                mime="application/octet-stream"
            )

            # (선택) 가공된 CSV 다운로드
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 전처리된 CSV 다운로드",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
            
    if st.button("처음부터 다시 하기"):
        st.session_state.clear()
        st.rerun()