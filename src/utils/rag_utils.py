import logging
import os
import tiktoken
from typing import Optional,Any, List ,Tuple
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from src.modules.create_answer import create_embedding
from src.utils.chatbot_utils import init_openai

MAX_TOKEN_COUNT_FOR_SOURCE_TEXT = 3000
logger = logging.getLogger(__name__)


def init_azure_search():
    """
    azureの検索サービスのクライアントを作成するよ
    Returns:
        search_client (SearchClient): azureの検索サービスのクライアント
    """
    assert (
        "AZURE_AI_SEARCH_ENDPOINT" in os.environ
    ), "AZURE_AI_SEARCH_ENDPOINT の環境変数がセットされていないよ"
    assert (
        "AZURE_AI_SEARCH_API_KEY" in os.environ
    ), "AZURE_AI_SEARCH_API_KEY の環境変数がセットされていないよ"
    assert (
        "AZURE_AI_SEARCH_INDEX_NAME" in os.environ
    ), "AZURE_AI_SEARCH_INDEX_NAME の環境変数がセットされていないよ"
    search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    search_credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=search_credential,
    )
    return search_client


def query_index_use_user_question(user_question: str):
    """
    貰った質問文に対して、azureの検索サービスを使って関連するドキュメントを返すよ
    Args:
        user_question (str): ユーザーからの質問文
    Returns:
        results (List[Dict[str, Any]]): 関連するドキュメント
    """
    search_client = init_azure_search()
    results = search_client.search(
        search_text=user_question,
        vector=Vector(
            value=create_embedding(user_question),  # ベクトルクエリ
            k=10,
            fields="query_target_vector",
        ),
        top=10,
    )
    return results


def format_query_results(query_results):
    """
    貰った関連するドキュメントを、
    ChatGPTのトークン数上限の限界まで繋げていき、あとはくっつけるだけの状態にするよ
    Args:
        query_results (List[Dict[str, Any]]): 関連するドキュメント
    Returns:
        source_text (str): １つにまとめた関連ドキュメント
    """    
    source_text = ""
    for i, result in enumerate(query_results):
        subject = result["query_target"]
        contents = result["contents"]

        if (
            calc_token_count(
                model=os.environ.get("OPENAI_CHAT_COMPLETION_MODEL", None),
                text=source_text
                + f"[{i}]:"
                + f"{i} subject: {subject}, contents: {contents}"
                + "\n",
            )
            > MAX_TOKEN_COUNT_FOR_SOURCE_TEXT
        ):
            break
        source_text += f"{i} subject: {subject}, contents: {contents}"

    return source_text


def calc_token_count(model: str, text: str) -> int:
    """
    貰ったテキストのトークン数を計算するよ
    Args:
        model (str): モデル名
        text (str): テキスト
    Returns:
        token_count (int): トークン数
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        logger.error(f"トークンカウントの計算中にエラーが発生しました in calc_token_count", exc_info=True)
        return 0


def create_response(message: Tuple[str,str]) -> str:
    """
    貰った質問文に対して、openaiのChatGPTを使って返答を生成するよ
    Args:
        message (Tuple[str,str]): ユーザーからの質問文
    Returns:
        response (str): ChatGPTの返答
    """
    openai_client = init_openai()
    try:
        response = openai_client.chat.completions.create(
            messages=message,
            model=os.environ.get("OPENAI_CHAT_COMPLETION_MODEL", "gpt-3.5-turbo"),
            )
        message = response.choices[0].message
        return message.content
    except Exception:
        logger.error("Error occurred while getting chat response.", exc_info=True)
        return None
    