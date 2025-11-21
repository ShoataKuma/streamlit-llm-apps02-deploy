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


def get_llm_response(user_input: str, expert_type: str, chat_history: list) -> str:
    """
    入力テキストと専門家タイプ、会話履歴を受け取り、LLMからの回答を返す関数
    
    Args:
        user_input (str): ユーザーからの入力テキスト
        expert_type (str): 専門家のタイプ（"healthcare" または "law"）
        chat_history (list): 会話履歴のリスト
    
    Returns:
        str: LLMからの回答
    """
    # LLMモデルの初期化（GPT-4oを使用）
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 専門家タイプに応じてシステムメッセージを設定
    if expert_type == "healthcare":
        system_content = "あなたはヘルスケアの専門家です。医療、健康、栄養、運動などに関する質問に専門的な知識を持って回答してください。"
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
    - **ヘルスケアの専門家**: 医療、健康、栄養、運動などに関する質問に回答
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
        options=["healthcare", "law"],
        format_func=lambda x: "👨‍⚕️ ヘルスケアの専門家" if x == "healthcare" else "⚖️ 日本の法律の専門家",
        index=0
    )
    
    # 選択された専門家の表示
    if expert_type == "healthcare":
        st.sidebar.success("現在の専門家: ヘルスケアの専門家 👨‍⚕️")
        st.sidebar.info("医療、健康、栄養、運動などに関する質問にお答えします。")
    else:
        st.sidebar.success("現在の専門家: 日本の法律の専門家 ⚖️")
        st.sidebar.info("日本の法律、法規制、法的手続きなどに関する質問にお答えします。")
    
    # 会話履歴のリセットボタン
    st.sidebar.divider()
    if st.sidebar.button("🔄 会話をリセット", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    # 会話履歴の表示
    st.sidebar.divider()
    st.sidebar.subheader("📊 会話統計")
    st.sidebar.metric("会話のやり取り数", len(st.session_state.messages) // 2)
    
    # メインコンテンツエリア
    st.header("💬 会話")
    
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
