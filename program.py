import streamlit as st
import pandas as pd
import numpy as np
import torch
import google.generativeai as genai
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import os


# --- [1. 환경 설정 및 API 연결] ---
st.set_page_config(page_title="Data Preprocessing Agent", layout="wide")

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
# model = genai.GenerativeModel('gemini-pro') 
model = genai.GenerativeModel('gemini-2.5-flash') # Or gemini-1.5-pro-latest


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
def get_agent_plan(df, goal, target_col):
    """Gemini가 전체 전처리 계획을 수립합니다."""
    sample_data = df.head(5).to_csv(index=False)
    null_info = df.isnull().mean().to_dict()
    column_types = df.dtypes.astype(str).to_dict()
    unique_counts = df.nunique().to_dict()
    
    prompt = f"""
    당신은 데이터 전처리 전문가 에이전트입니다.
    사용자의 목표: {goal}
    타겟 컬럼(예측 변수): {target_col} (이 컬럼은 삭제하지 말고, 적절한 인코딩이나 처리를 제안하세요.)
    데이터 샘플: {sample_data}
    결측률: {null_info}
    타입: {column_types}
    Unique 값 개수: {unique_counts}
    
    각 컬럼별로 [Drop, Fill_Median, Fill_Mode, Fill_Zero, Normalize, Encode_OneHot, Encode_Label, Pass] 중 하나를 선택해 단계별 계획을 세우세요.
    
    규칙:
    1. 범주형 데이터(object)는 반드시 Encode_OneHot(카테고리 수 적을 때) 또는 Encode_Label(많을 때)을 수행해야 합니다.
    2. 타겟 컬럼인 '{target_col}'은 절대 Drop하지 마세요. 범주형이면 Encode_Label, 수치형이면 Pass나 Normalize를 추천하세요.
    3. JSON 리스트 형식으로만 응답하세요.
    
    [
        {{"col": "컬럼명", "action": "선택한 액션", "reason": "이유(한글)"}}
    ]
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error calling Gemini: {e}") 
        return [{"col": c, "action": "Pass", "reason": f"API 오류로 기본 Pass 처리 ({str(e)})"} for c in df.columns]

def apply_step(df, col, action):
    """단일 스텝(컬럼 액션)을 수행하고 결과 데이터프레임을 반환합니다."""
    new_df = df.copy()
    
    if col not in new_df.columns:
        return new_df

    try:
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
                data = new_df[[col]].values
                new_df[col] = scaler.fit_transform(data).flatten()
        elif action == "Encode_Label":
            le = LabelEncoder()
            # 결측치는 임시 처리 후 인코딩
            new_df[col] = new_df[col].fillna("Unknown").astype(str)
            new_df[col] = le.fit_transform(new_df[col])
        elif action == "Encode_OneHot":
            # One-Hot은 컬럼이 늘어나므로 처리가 조금 다릅니다.
            # 하지만 여기서는 해당 컬럼을 원핫 인코딩한 데이터프레임으로 교체합니다.
            dummies = pd.get_dummies(new_df[col], prefix=col, drop_first=False)
            new_df = pd.concat([new_df.drop(columns=[col]), dummies], axis=1)
            
    except Exception as e:
        st.error(f"Action '{action}' failed on '{col}': {e}")
            
    return new_df

def plot_comparison(old_df, new_df, col):
    """변경 전후 분포 비교 시각화 (Plotly)"""
    # 1. 컬럼이 사라진 경우 (Drop, OneHot 등)
    if col not in new_df.columns:
        st.info(f"ℹ️ '{col}' 컬럼은 처리 후 구조가 변경되었거나 삭제되었습니다. (예: One-Hot Encoding)")
        return

    col_c1, col_c2 = st.columns(2)
    
    # 2. 수치형 데이터 시각화
    if pd.api.types.is_numeric_dtype(new_df[col]):
        with col_c1:
            # Histogram
            fig = go.Figure()
            # Original
            fig.add_trace(go.Histogram(x=old_df[col].dropna(), name='Original', opacity=0.5, marker_color='gray'))
            # Transformed
            fig.add_trace(go.Histogram(x=new_df[col], name='Transformed', opacity=0.5, marker_color='blue'))
            fig.update_layout(title_text=f"{col} Distribution (Histogram)", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_c2:
            # Box Plot
            fig2 = go.Figure()
            fig2.add_trace(go.Box(y=old_df[col].dropna(), name='Original', marker_color='gray'))
            fig2.add_trace(go.Box(y=new_df[col], name='Transformed', marker_color='blue'))
            fig2.update_layout(title_text=f"{col} Box Plot (Outliers)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 🔢 통계 요약")
        desc_old = old_df[col].describe()
        desc_new = new_df[col].describe()
        stats_df = pd.DataFrame({'Original': desc_old, 'Transformed': desc_new})
        st.dataframe(stats_df.T, use_container_width=True)

    # 3. 범주형 데이터 시각화
    else:
        # 상위 10개 카테고리
        top_n = 10
        old_counts = old_df[col].value_counts().head(top_n)
        new_counts = new_df[col].value_counts().head(top_n)
        
        with col_c1:
            fig = px.bar(x=old_counts.index, y=old_counts.values, title=f"Original Top {top_n}", labels={'x':'Category', 'y':'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col_c2:
            fig2 = px.bar(x=new_counts.index, y=new_counts.values, title=f"Transformed Top {top_n}", labels={'x':'Category', 'y':'Count'})
            st.plotly_chart(fig2, use_container_width=True)

# --- [3. 메인 로직] ---
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
    st.session_state.current_step_idx = 0
    st.session_state.log = [] 

st.title("🤖 Advanced AI Data Preprocessing Agent")

# [Step 1] 데이터 업로드 & 목표 & 타겟 설정
if st.session_state.step == 'upload':
    st.subheader("1. 데이터 & 목표 설정")
    uploaded_file = st.file_uploader("CSV 데이터를 업로드하세요.", type="csv")
    user_goal = st.text_input("분석 목표 (예: 타이타닉 생존자 예측)", placeholder="이 데이터로 무엇을 하고 싶으신가요?")
    
    if uploaded_file:
        # 임시로 읽어서 컬럼 목록 보여주기
        temp_df = pd.read_csv(uploaded_file)
        st.write("데이터 미리보기:")
        st.dataframe(temp_df.head(3))
        
        target_col = st.selectbox("🎯 타겟 컬럼(예측할 정답)을 선택하세요:", temp_df.columns)
        
        if user_goal and st.button("AI에게 계획 요청하기 🚀"):
            st.session_state.original_df = temp_df
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.goal = user_goal
            st.session_state.target_col = target_col
            st.session_state.history = []
            
            with st.spinner("Gemini가 데이터를 분석하고 전처리 전략을 수립 중입니다..."):
                plan = get_agent_plan(st.session_state.df, user_goal, target_col)
                st.session_state.plan = plan
                st.session_state.step = 'plan_edit' # 새로운 단계
                st.rerun()

# [Step 2] 계획 검토 및 수정 (Plan Editor)
elif st.session_state.step == 'plan_edit':
    st.subheader("2. AI 제안 전처리 계획 검토")
    st.info("AI가 제안한 계획입니다. 마음에 들지 않으면 수정할 수 있습니다.")
    
    # 딕셔너리 리스트 -> DataFrame 변환
    plan_df = pd.DataFrame(st.session_state.plan)
    
    # 수정 가능한 Data Editor
    edited_plan_df = st.data_editor(
        plan_df,
        column_config={
            "col": st.column_config.TextColumn("컬럼명", disabled=True),
            "action": st.column_config.SelectboxColumn(
                "액션",
                options=["Pass", "Drop", "Fill_Median", "Fill_Mode", "Fill_Zero", "Normalize", "Encode_Label", "Encode_OneHot"],
                required=True
            ),
            "reason": st.column_config.TextColumn("이유 (AI 생성)", disabled=True)
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed"
    )
    
    col1, col2 = st.columns(2)
    if col1.button("이대로 실행하기 ▶️", type="primary"):
        # 수정된 내용을 다시 리스트로 변환
        st.session_state.plan = edited_plan_df.to_dict('records')
        st.session_state.step = 'execute_loop'
        st.rerun()
        
    if col2.button("처음으로 돌아가기"):
        st.session_state.step = 'upload'
        st.rerun()

# [Step 3] 실행 루프
elif st.session_state.step == 'execute_loop':
    plan = st.session_state.plan
    idx = st.session_state.current_step_idx
    
    if idx < len(plan):
        current_item = plan[idx]
        col = current_item['col']
        action = current_item['action']
        reason = current_item['reason']
        
        # 이미 처리 과정에서 컬럼이 사라졌을 수도 있음 (예: 이전 단계의 OneHot 등)
        # 하지만 원본 컬럼명 기준 루프이므로, 현재 df에 col이 있는지 체크 필요
        col_exists = col in st.session_state.df.columns
        
        progress = (idx / len(plan))
        st.progress(progress, text=f"Processing... ({int(progress*100)}%)")
        
        st.subheader(f"Step {idx+1}/{len(plan)}: {col}")
        
        if not col_exists:
            st.warning(f"⚠️ 컬럼 '{col}'을(를) 찾을 수 없습니다. (이미 삭제되었거나 변형됨)")
            # 자동 스킵
            if st.button("다음으로 넘어가기"):
                st.session_state.log.append(f"Step {idx+1}: {col} -> Skipped (Not Found)")
                st.session_state.current_step_idx += 1
                st.rerun()
            st.stop()

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**{col}** → **`{action}`**")
            st.caption(f"Reason: {reason}")

        # Preview
        preview_df = apply_step(st.session_state.df, col, action)
        
        with st.expander("🔍 결과 미리보기 (Interactive Chart)", expanded=True):
            if action != "Drop":
                plot_comparison(st.session_state.df, preview_df, col)
            else:
                st.error(f"🗑️ '{col}' 컬럼이 제거됩니다.")

        st.write("---")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ 승인 (Apply)", type="primary", use_container_width=True):
            st.session_state.history.append(st.session_state.df.copy())
            st.session_state.df = preview_df
            st.session_state.log.append(f"{col}: {action}")
            st.session_state.current_step_idx += 1
            st.rerun()
            
        if c2.button("❌ 건너뛰기 (Pass)", use_container_width=True):
            st.session_state.history.append(st.session_state.df.copy())
            # df 변경 없음
            st.session_state.log.append(f"{col}: Pass (User Skipped)")
            st.session_state.current_step_idx += 1
            st.rerun()
            
        if c3.button("↩️ 실행 취소 (Undo)", use_container_width=True):
            if st.session_state.history:
                st.session_state.df = st.session_state.history.pop()
                if st.session_state.log: st.session_state.log.pop()
                st.session_state.current_step_idx -= 1
                st.rerun()
            else:
                st.warning("돌아갈 단계가 없습니다.")

    else:
        st.session_state.step = 'final'
        st.rerun()

# [Step 4] 완료 및 다운로드
elif st.session_state.step == 'final':
    st.balloons()
    st.success("🎉 모든 전처리가 완료되었습니다!")
    
    final_df = st.session_state.df
    target_col = st.session_state.target_col
    
    st.subheader("📊 최종 데이터 요약")
    st.dataframe(final_df.head())
    st.write(f"Shape: {final_df.shape}")
    
    with st.expander("�️ 처리 로그"):
        for l in st.session_state.log:
            st.text(l)
            
    # Tensor 생성 및 다운로드
    if st.button("Generate PyTorch Tensors"):
        # 1. Target 분리 attempt
        if target_col in final_df.columns:
            # 타겟이 변형되지 않았거나 LabelEncoding된 상태
            y = final_df[target_col]
            X = final_df.drop(columns=[target_col])
        else:
            # 타겟 컬럼이 OneHot 등으로 이름이 바뀌었거나 사라졌을 수 있음
            # 단순화를 위해 만약 타겟이 없으면 가장 마지막 컬럼을 타겟으로 가정하거나,
            # OneHot된 컬럼들을 찾아서 y로 묶어줘야 함.
            # 지금은 간단히 경고 후 전체를 X로.
            st.warning(f"타겟 컬럼 '{target_col}'이(가) 보이지 않습니다. One-Hot Encoding 되었을 수 있습니다.")
            # 이름에 target_col이 포함된 컬럼들을 y로 간주 (간이 로직)
            target_cols = [c for c in final_df.columns if str(c).startswith(f"{target_col}_")]
            if target_cols:
                y = final_df[target_cols]
                X = final_df.drop(columns=target_cols)
                st.info(f"타겟으로 추정되는 컬럼들: {target_cols}")
            else:
                st.error("타겟 컬럼을 찾을 수 없어 전체를 X로 사용합니다.")
                X = final_df
                y = pd.Series(np.zeros(len(X))) # Dummy y

        # 2. 모두 숫자인지 확인
        try:
            # 텐서 변환을 위해 object 타입 등이 남아있으면 강제 변환 시도 or 에러
            # 여기서 numeric_only=True를 하면 데이터 유실됨. 
            # 앞단계에서 Encoding을 강제했으므로, 여기선 coerce로 변환 시도
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # y 처리
            if isinstance(y, pd.DataFrame): # OneHot된 target
                 y = y.apply(pd.to_numeric, errors='coerce').fillna(0).values
            else:
                 y = pd.to_numeric(y, errors='coerce').fillna(0).values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
            
            st.write(f"X_train: {X_train_t.shape}, y_train: {y_train_t.shape}")
            
            # Save
            buffer = io.BytesIO()
            torch.save({
                'X_train': X_train_t, 'X_test': X_test_t,
                'y_train': y_train_t, 'y_test': y_test_t,
                'feature_names': list(X.columns)
            }, buffer)
            
            st.download_button("💾 Download .pt file", buffer.getvalue(), "data.pt")
            
        except Exception as e:
            st.error(f"텐서 변환 중 오류 발생: {e}")
            st.caption("모든 컬럼이 숫자형으로 변환되었는지 확인하세요.")

    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()