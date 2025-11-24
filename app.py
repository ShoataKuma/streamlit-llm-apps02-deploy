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
    elif expert_type == "concierge":
        system_content = """あなたは株式会社隈川写真館の優秀なコンシェルジュです。お客様のご質問に対して、当館のサービスや情報を基に丁寧にお答えします。

【隈川写真館の情報】

■ 基本情報
- 1940年創業の歴史ある写真館
- 所在地：埼玉県富士見市鶴瀬東1-8-20
- 電話：049-251-0476
- 営業時間：10:00～18:00
- 定休日：木曜日
- Webサイト：https://kumakawa.co.jp 、 https://kumakawa.co.jp/satellite/

■ 撮影メニュー
- ベビーアルバム
- デザインアルバム
- お誕生日記念
- 家族写真・結婚記念
- お宮参り
- 初節句
- 入園・入学
- 七五三
- 卒園・卒業
- 成人式
- 証明写真（通常・プレミアム・オーディションフォト）
- 学校スナップ写真
- 卒業アルバム制作

■ 撮影の流れ
1. ご予約：完全予約制（成人の日のみご来館順）
2. 衣装選び：七五三、成人男性袴、卒業袴は撮影日前に選ぶ
3. 撮影：5分～30分（デザインアルバムは1時間弱）
4. できあがり：約1ヶ月後（デザインアルバムは約6週間後）

■ 特徴
- アットホームな雰囲気で撮影
- お子様目線の雰囲気作り
- 豊富なレンタル衣装
- 駐車場あり（通常2台、繁忙期4台）
- 現金・カード・電子マネー決済可能
- データ販売あり
- 携帯待受画像プレゼント

■ アルバム会員
- 登録月に撮影、30％～10％OFF

■ キャンセルポリシー
- 体調不良等のやむを得ない理由：お日にち変更1回まで無料
- 美容師手配済みの場合：1ヶ月前からキャンセル料20％～100％
- 天気や自己都合によるキャンセル：キャンセル料あり

【回答のガイドライン】
- 上記の情報を基に、お客様の質問に正確にお答えする
- 丁寧で親しみやすい言葉遣いを使う
- 詳細情報が必要な場合は、Webサイトやお電話でのお問い合わせをご案内する
- お客様のニーズに合わせて最適なプランを提案する
- 常にお客様目線で、親身に対応する"""
    elif expert_type == "law":
        system_content = "あなたは日本の法律の専門家です。日本の法律、法規制、法的手続きなどに関する質問に専門的な知識を持って回答してください。"
    elif expert_type == "legal_check":
        system_content = """あなたは日本の法律に詳しいリーガルチェックの専門家です。
ユーザーから与えられた文書や契約書、規約、利用規約などの法的妥当性をチェックし、改善案を提供します。

【チェック項目】
- 法令違反の有無
- 不当条項の有無
- 消費者契約法、特定商取引法、個人情報保護法などへの適合性
- 曖昧な表現や誤解を招く可能性のある記述
- リスクのある条項

【重要な指示】
- 改善後の文書は、元の文書の内容を**すべて網羅**すること
- 元の文書にあった条項や内容を省略せず、すべて含めること
- 問題のない部分もそのまま含め、問題のある部分のみを修正すること
- 元の文書の構造（条項番号、見出しなど）を維持すること
- 短い文書でも長い文書でも、必ず完全な全文を生成すること

【回答形式】
必ず以下の形式で回答してください：

## リーガルチェック結果

### 総合評価
[適法 / 要注意 / 問題あり] のいずれかを明記

### 問題点
1. [問題点の詳細]
2. [問題点の詳細]
（問題がない場合は「特に問題はありません」と記載）

### 改善提案
[改善が必要な場合のみ、具体的な改善案を記載]

---
## 改善後の文書

[元の文書の内容をすべて網羅し、問題点を修正した完全版を記載]
[元の文書が短い場合でも、省略せず全文を記載すること]
[改善が不要な場合は「原文のままで問題ありません」と記載]

---

※ このチェックは一般的なアドバイスであり、最終的な法的判断は弁護士にご相談ください。"""
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
    - **隈川写真館コンシェルジュ**: 隈川写真館のサービス、料金、撮影の流れなどについて回答
    - **日本の法律の専門家**: 日本の法律、法規制、法的手続きなどに関する質問に回答
    - **リーガルチェック**: 文書の法的妃当性をチェックし、改善後の文書を生成
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
        options=["business", "concierge", "law", "legal_check"],
        format_func=lambda x: "💼 優秀で丁寧な営業マン" if x == "business" else ("🏪 隈川写真館コンシェルジュ" if x == "concierge" else ("⚖️ 日本の法律の専門家" if x == "law" else "📝 リーガルチェック")),
        index=0
    )
    
    # 選択された専門家の表示
    if expert_type == "business":
        st.sidebar.success("現在の専門家: 優秀で丁寧な営業マン 💼")
        st.sidebar.info("文章をビジネス文書として整え、読みやすく洗練された表現に変換します。")
    elif expert_type == "concierge":
        st.sidebar.success("現在の専門家: 隈川写真館コンシェルジュ 🏪")
        st.sidebar.info("隈川写真館のサービスや情報についてお答えします。")
    elif expert_type == "law":
        st.sidebar.success("現在の専門家: 日本の法律の専門家 ⚖️")
        st.sidebar.info("日本の法律、法規制、法的手続きなどに関する質問にお答えします。")
    else:
        st.sidebar.success("現在の専門家: リーガルチェック 📝")
        st.sidebar.info("文書の法的妃当性をチェックし、改善案を提供します。")
    
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
    
    # 会話履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ビジネスメールのまとめを表示（生成された場合）- 会話の最後に表示
    if "summary_email" in st.session_state and st.session_state.summary_email:
        st.divider()
        st.success("✅ ビジネスメールが生成されました！")
        with st.expander("📧 生成されたビジネスメール", expanded=True):
            st.markdown(st.session_state.summary_email)
            
            # コピー用のテキストエリア
            st.text_area(
                "コピー用",
                st.session_state.summary_email,
                height=300,
                key="summary_copy"
            )
        
        # コピーボタンとまとめを閉じるボタンを横並びに配置
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 コピー", use_container_width=True):
                st.toast("クリップボードにコピーされました！", icon="✅")
                # JavaScriptを使用してクリップボードにコピー
                st.write(f"""
                <script>
                navigator.clipboard.writeText(`{st.session_state.summary_email.replace('`', '\\`')}`);
                </script>
                """, unsafe_allow_html=True)
        with col2:
            if st.button("❌ まとめを閉じる", use_container_width=True):
                del st.session_state.summary_email
                st.rerun()
    
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
                    
                    # リーガルチェックの場合、改善後の文書を抽出してコピーボタンを表示
                    if expert_type == "legal_check" and "## 改善後の文書" in response:
                        # 改善後の文書を抽出
                        parts = response.split("## 改善後の文書")
                        if len(parts) > 1:
                            improved_text = parts[1].split("---")[0].strip()
                            if improved_text and "原文のままで問題ありません" not in improved_text:
                                st.divider()
                                st.subheader("📋 改善後の文書をコピー")
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.text_area(
                                        "改善後の文書",
                                        improved_text,
                                        height=200,
                                        key=f"improved_text_{len(st.session_state.messages)}"
                                    )
                                with col2:
                                    st.write("")
                                    st.write("")
                                    if st.button("📋 コピー", key=f"copy_btn_{len(st.session_state.messages)}", use_container_width=True):
                                        st.toast("クリップボードにコピーされました！", icon="✅")
                                        st.write(f"""
                                        <script>
                                        navigator.clipboard.writeText(`{improved_text.replace('`', '\\`')}`);
                                        </script>
                                        """, unsafe_allow_html=True)
                    
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
