"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langsmith import Client
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


PROMPT_IDENTIFIER = "leonanluppi/bug_to_user_story_v1"
OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"

def extract_messages(prompt) -> dict:
    """
    Extrai system_prompt e user_prompt de um prompt do LangChain.

    O Hub devolve um ChatPromptTemplate na maioria dos casos, mas prompts
    antigos podem vir como PromptTemplate simples. Os dois formatos são
    tratados aqui para o YAML sair sempre com a mesma estrutura.

    Args:
        prompt: Objeto retornado por hub.pull()

    Returns:
        Dicionário com as chaves system_prompt e user_prompt
    """
    messages = getattr(prompt, 'messages', None)

    if messages is None:
        return {
            'system_prompt': getattr(prompt, 'template', str(prompt)),
            'user_prompt': ''
        }

    system_parts = []
    user_parts = []

    for message in messages:
        template = getattr(getattr(message, 'prompt', None), 'template', None)
        if template is None:
            continue

        role = message.__class__.__name__.lower()
        if 'system' in role:
            system_parts.append(template)
        else:
            user_parts.append(template)

    return {
        'system_prompt': "\n\n".join(system_parts),
        'user_prompt': "\n\n".join(user_parts)
    }

def get_prompt_metadata(identifier: str) -> dict:
    """
    Busca descrição e tags do prompt no LangSmith.

    Os metadados são opcionais: se a consulta falhar, o pull continua
    apenas com o conteúdo do prompt.

    Args:
        identifier: Identificador no formato owner/nome

    Returns:
        Dicionário com description e tags (vazios se indisponíveis)
    """
    try:
        info = Client().get_prompt(identifier)
        return {
            'description': info.description or "",
            'tags': list(info.tags or [])
        }
    except Exception as e:
        print(f"Não foi possível ler os metadados do prompt: {e}")
        return {'description': "", 'tags': []}

def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull do prompt de baixa qualidade e salva em YAML local.

    Returns:
    True se sucesso, False caso contrário
    """

    try:
        prompt = hub.pull(PROMPT_IDENTIFIER)
    except Exception as e:
        print(f"Erro ao puxar o prompt: {e}")
        return False

    metadata = get_prompt_metadata(PROMPT_IDENTIFIER)
    content = extract_messages(prompt)

    prompt_key = PROMPT_IDENTIFIER.split("/")[-1]
    data = {
        prompt_key: {
            'description': metadata['description'] or "Prompt para converter relatos de bugs em User Stories",
            'system_prompt': content['system_prompt'],
            'user_prompt': content['user_prompt'],
            'version': "v1",
            'source': PROMPT_IDENTIFIER,
            'input_variables': list(prompt.input_variables),
            'tags': metadata['tags'] or ["bug-analysis", "user-story", "product-management"]
        }
    }

    if not save_yaml(data, OUTPUT_PATH):
        return False

    print(f"Prompt '{PROMPT_IDENTIFIER}' salvo em {OUTPUT_PATH}")
    return True

def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    if not pull_prompts_from_langsmith():
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())