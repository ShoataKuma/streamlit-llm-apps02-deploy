import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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


def get_llm_response(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、LLMからの回答を返す関数
    
    Args:
        user_input (str): ユーザーからの入力テキスト
        expert_type (str): 専門家のタイプ（"healthcare" または "law"）
    
    Returns:
        str: LLMからの回答
    """
    # LLMモデルの初期化
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # 専門家タイプに応じてシステムメッセージを設定
    if expert_type == "healthcare":
        system_content = "あなたはヘルスケアの専門家です。医療、健康、栄養、運動などに関する質問に専門的な知識を持って回答してください。"
    elif expert_type == "law":
        system_content = "あなたは日本の法律の専門家です。日本の法律、法規制、法的手続きなどに関する質問に専門的な知識を持って回答してください。"
    else:
        system_content = "You are a helpful assistant."
    
    # メッセージの作成
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]
    
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
    LangChainとOpenAIのGPT-4o-miniモデルを使用して、選択した専門家として回答します。
    
    ## 🔧 操作方法
    1. **専門家を選択**: 下のラジオボタンから相談したい専門家を選択してください
    2. **質問を入力**: テキストエリアに質問や相談内容を入力してください
    3. **送信**: 「送信」ボタンをクリックすると、選択した専門家としてAIが回答します
    
    ### 利用可能な専門家
    - **ヘルスケアの専門家**: 医療、健康、栄養、運動などに関する質問に回答
    - **日本の法律の専門家**: 日本の法律、法規制、法的手続きなどに関する質問に回答
    """)
    
    st.divider()
    
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
    
    # メインコンテンツエリア
    st.header("💬 質問を入力してください")
    
    # 入力フォーム
    user_input = st.text_area(
        "質問内容:",
        height=150,
        placeholder="例: 健康的な食事のポイントを教えてください（ヘルスケア専門家の場合）\n例: 契約書の有効期限について教えてください（法律専門家の場合）"
    )
    
    # 送信ボタン
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.button("🚀 送信", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ クリア", use_container_width=True)
    
    # クリアボタンが押された場合
    if clear_button:
        st.rerun()
    
    # 送信ボタンが押された場合の処理
    if submit_button:
        if user_input.strip():
            with st.spinner("🤔 考え中..."):
                try:
                    # LLMから回答を取得
                    response = get_llm_response(user_input, expert_type)
                    
                    # 回答を表示
                    st.success("回答が生成されました！")
                    st.subheader("📝 回答:")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    st.info("OpenAI APIキーが正しく設定されているか確認してください。")
        else:
            st.warning("⚠️ 質問内容を入力してください。")
    
    # フッター
    st.divider()
    st.caption("Powered by LangChain and OpenAI GPT-4o-mini")


if __name__ == "__main__":
    main()
