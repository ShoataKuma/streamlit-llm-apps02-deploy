import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# .envファイルから環境変数を読み込む
load_dotenv()

# Streamlit Cloudの場合はst.secretsから環境変数を設定
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ページ設定
st.set_page_config(
    page_title="専門家AI チャットアプリ",
    page_icon="🤖",
    layout="wide"
)


def generate_summary_email(messages: list) -> str:
    """
    会話履歴からビジネスメール形式のまとめを生成する関数
    
    Args:
        messages (list): 会話履歴のリスト
    
    Returns:
        str: ビジネスメール形式のまとめ
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 会話履歴をテキストに変換
    conversation_text = ""
    for msg in messages:
        role = "質問" if msg["role"] == "user" else "回答"
        conversation_text += f"{role}: {msg['content']}\n\n"
    
    # サマリー生成用のプロンプト
    system_content = """あなたは優秀で丁寧な営業マンです。卒業アルバム制作業者として、学校の先生方とやり取りをしています。ビジネスマンとしてのルールやマナーに則り、親切丁寧に、
以下の会話履歴の内容を、1つのビジネスメールとして分かりやすくまとめてください。

【重要な指示】
- 書き出しは必ず「平素より格別のご高配を賜り厚く御礼申し上げます。」から始める
- 会話の内容を論理的に整理し、ビジネス文書として適切な形式にする
- 敬語を適切に使用し、丁寧で分かりやすい表現を使う
- 適切な段落分けと箇条書きを使用して読みやすくする
- 催促や否定をする場合は、必ず先生方を想った表現にする
  例：「卒業アルバム作成スケジュールの都合上、～をできるだけ早めにお願いいたします」
  例：「これ以上遅くなってしまいますと、校正閲覧時間が短くなるなど、先生方にとってデメリットが大きくなってしまうため」
- 常に先生方の立場に立ち、先生方のためという姿勢を明確にする
- 最後は適切な締めの言葉で終わる
- メールの最後に以下の署名を必ず追加する：

────────────────────
株式会社隈川写真館
担当：隈川
TEL: 049-251-0476
Email: h@kumakawa.co.jp
────────────────────"""
    
    messages_for_llm = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"以下の会話履歴をビジネスメールとしてまとめてください:\n\n{conversation_text}")
    ]
    
    result = llm.invoke(messages_for_llm)
    return result.content


def get_llm_response(user_input: str, expert_type: str, chat_history: list) -> str:
    """
    入力テキストと専門家タイプ、会話履歴を受け取り、LLMからの回答を返す関数
    
    Args:
        user_input (str): ユーザーからの入力テキスト
        expert_type (str): 専門家のタイプ（"business" または "law"）
        chat_history (list): 会話履歴のリスト
    
    Returns:
        str: LLMからの回答
    """
    # LLMモデルの初期化（GPT-4oを使用）
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 専門家タイプに応じてシステムメッセージを設定
    if expert_type == "business":
        system_content = """あなたは優秀で丁寧な営業マンです。株式会社隈川写真館の担当者「隈川」として、卒業アルバム制作業者として学校の先生方とやり取りをしています。

ユーザーから与えられた文章をビジネス文書として整え、読みやすく洗練された文章に変換してください。

【重要な指示】
- 文章の話者（書き手）は常に「隈川」とし、一人称で書く
- 「私」や「弊社」という表現を使う場合も、文脈から「隈川」が書いていることが明確にわかるようにする
- 敬語を適切に使用し、ビジネスシーンで使える丁寧で分かりやすい表現にする
- 催促や否定をする場合は、相手（先生方）を想った表現を使用する
- 催促の例：「卒業アルバム作成スケジュールの都合上、○月○日までにご確認いただく必要がございます」
- デメリット説明の例：「これ以上お時間をいただきますと、校正閲覧時間が短くなるなど、先生方にとってデメリットが大きくなってしまうため」
- 常に先生方の立場に立ち、先生方の利益やメリットを考慮した文章構成にする
- 催促や否定も、あくまで先生方のためという姿勢を明確にする
- 相手への配慮と敬意を忘れず、丁寧でありながら要点を明確に伝える"""
    elif expert_type == "law":
        system_content = "あなたは日本の法律の専門家です。日本の法律、法規制、法的手続きなどに関する質問に専門的な知識を持って回答してください。"
    else:
        system_content = "You are a helpful assistant."
    
    # メッセージの作成（システムメッセージ + 会話履歴 + 新しい質問）
    messages = [SystemMessage(content=system_content)]
    
    # 会話履歴を追加
    messages.extend(chat_history)
    
    # 新しい質問を追加
    messages.append(HumanMessage(content=user_input))
    
    # LLMに問い合わせ
    result = llm.invoke(messages)
    
    return result.content


def main():
    # アプリケーションのタイトル
    st.title("🤖 専門家AI チャットアプリ")
    
    # アプリケーションの説明
    st.markdown("""
    ## 📋 アプリケーション概要
    このアプリケーションは、専門分野に特化したAIアシスタントと対話できるツールです。
    LangChainとOpenAIのGPT-4oモデルを使用して、選択した専門家として回答します。
    会話履歴を保持するため、前の質問を深掘りした質問も可能です。
    
    ## 🔧 操作方法
    1. **専門家を選択**: 下のラジオボタンから相談したい専門家を選択してください
    2. **質問を入力**: テキストエリアに質問や相談内容を入力してください
    3. **送信**: 「送信」ボタンをクリックすると、選択した専門家としてAIが回答します
    4. **会話履歴**: 過去の会話は保持され、続けて質問できます
    5. **リセット**: 「会話をリセット」ボタンで会話履歴をクリアできます
    
    ### 利用可能な専門家
    - **優秀で丁寧な営業マン**: 与えられた文章をビジネス文書として整え、読みやすく洗練された文章に変換
    - **日本の法律の専門家**: 日本の法律、法規制、法的手続きなどに関する質問に回答
    """)
    
    st.divider()
    
    # セッション状態の初期化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # サイドバーに専門家選択のラジオボタンを配置
    st.sidebar.header("専門家の選択")
    expert_type = st.sidebar.radio(
        "相談したい専門家を選択してください:",
        options=["business", "law"],
        format_func=lambda x: "💼 優秀で丁寧な営業マン" if x == "business" else "⚖️ 日本の法律の専門家",
        index=0
    )
    
    # 選択された専門家の表示
    if expert_type == "business":
        st.sidebar.success("現在の専門家: 優秀で丁寧な営業マン 💼")
        st.sidebar.info("文章をビジネス文書として整え、読みやすく洗練された表現に変換します。")
    else:
        st.sidebar.success("現在の専門家: 日本の法律の専門家 ⚖️")
        st.sidebar.info("日本の法律、法規制、法的手続きなどに関する質問にお答えします。")
    
    # 会話履歴のリセットボタン
    st.sidebar.divider()
    if st.sidebar.button("🔄 会話をリセット", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    # 営業マン選択時のまとめ生成ボタン
    if expert_type == "business" and len(st.session_state.messages) > 0:
        st.sidebar.divider()
        if st.sidebar.button("📧 ビジネスメールとしてまとめる", type="primary", use_container_width=True):
            with st.spinner("📝 ビジネスメールを生成中..."):
                try:
                    summary_email = generate_summary_email(st.session_state.messages)
                    st.session_state.summary_email = summary_email
                except Exception as e:
                    st.sidebar.error(f"エラー: {str(e)}")
    
    # 会話履歴の表示
    st.sidebar.divider()
    st.sidebar.subheader("📊 会話統計")
    st.sidebar.metric("会話のやり取り数", len(st.session_state.messages) // 2)
    
    # メインコンテンツエリア
    st.header("💬 会話")
    
    # ビジネスメールのまとめを表示（生成された場合）
    if "summary_email" in st.session_state and st.session_state.summary_email:
        st.success("✅ ビジネスメールが生成されました！")
        with st.expander("📧 生成されたビジネスメール", expanded=True):
            st.markdown(st.session_state.summary_email)
            
            # コピーボタン用のテキストエリアとコピーボタン
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text_area(
                    "コピー用",
                    st.session_state.summary_email,
                    height=300,
                    key="summary_copy"
                )
            with col2:
                st.write("")  # 上部の余白調整
                st.write("")
                if st.button("📋 コピー", use_container_width=True):
                    st.toast("クリップボードにコピーされました！", icon="✅")
                    # JavaScriptを使用してクリップボードにコピー
                    st.write(f"""
                    <script>
                    navigator.clipboard.writeText(`{st.session_state.summary_email.replace('`', '\\`')}`);
                    </script>
                    """, unsafe_allow_html=True)
        
        # まとめをクリア
        if st.button("❌ まとめを閉じる"):
            del st.session_state.summary_email
            st.rerun()
        
        st.divider()
    
    # 会話履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 入力フォーム
    user_input = st.chat_input("質問を入力してください...")
    
    # ユーザーが質問を送信した場合
    if user_input:
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # メッセージ履歴に追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AIの回答を生成
        with st.chat_message("assistant"):
            with st.spinner("🤔 考え中..."):
                try:
                    # LLMから回答を取得
                    response = get_llm_response(user_input, expert_type, st.session_state.chat_history)
                    
                    # 回答を表示
                    st.markdown(response)
                    
                    # メッセージ履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # チャット履歴に追加（LangChainのメッセージ形式）
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    
                except Exception as e:
                    error_msg = f"エラーが発生しました: {str(e)}"
                    st.error(error_msg)
                    st.info("OpenAI APIキーが正しく設定されているか確認してください。")
    
    # フッター
    st.divider()
    st.caption("Powered by LangChain and OpenAI GPT-4o")


if __name__ == "__main__":
    main()
