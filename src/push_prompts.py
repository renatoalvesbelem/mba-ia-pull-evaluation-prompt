"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()
PROMPT_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"
EXPECTED_VARIABLES = {"bug_report"}

def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    """
    Monta o ChatPromptTemplate a partir dos campos do YAML.

    Args:
        prompt_data: Dados do prompt

    Returns:
        ChatPromptTemplate com as mensagens de system e user
    """
    return ChatPromptTemplate.from_messages([
        ("system", prompt_data["system_prompt"]),
        ("human", prompt_data["user_prompt"])
    ])

def build_readme(prompt_data: dict) -> str:
    """
        Gera o readme publicado junto do prompt no Hub.

    Args:
        prompt_data: Dados do prompt

    Returns:
        Texto em Markdown com descrição e técnicas aplicadas
    """
    techniques = "\n".join(f"- {t}" for t in prompt_data.get("techniques_applied", []))

    return (
        f"# {PROMPT_KEY}\n\n"
        f"{prompt_data.get('description', '')}\n\n"
        f"## Técnicas de Prompt Engineering aplicadas\n\n"
        f"{techniques}\n\n"
        f"## Entrada\n\n"
        f"Variável `bug_report`: relato de bug em texto livre, vindo de usuário, suporte ou QA.\n\n"
        f"## Saída\n\n"
        f"User story em Markdown, com critérios de aceitação em Dado/Quando/Então e profundidade "
        f"proporcional ao nível de detalhe do relato.\n"
    )

def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    for field in ("description", "system_prompt", "user_prompt", "version"):
        if not str(prompt_data.get(field, "")).strip():
            errors.append(f"Campo obrigatório ausente ou vazio: {field}")

    system_prompt = str(prompt_data.get("system_prompt", ""))
    if "TODO" in system_prompt or "TODO" in str(prompt_data.get("user_prompt", "")):
        errors.append("O prompt ainda contém marcações TODO")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    if not errors:
        try:
            variables = set(build_chat_prompt(prompt_data).input_variables)
            if variables != EXPECTED_VARIABLES:
                errors.append(
                    f"Variáveis do template: {sorted(variables) or 'nenhuma'}. "
                    f"Esperado exatamente: {sorted(EXPECTED_VARIABLES)}"
                )
        except Exception as e:
            errors.append(f"Não foi possível montar o template: {e}")

    return (len(errors) == 0, errors)

def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    print(f"Publicando: {prompt_name}")

    try:
        url = hub.push(
            prompt_name,
            build_chat_prompt(prompt_data),
            new_repo_is_public=True,
            new_repo_description=prompt_data.get("description", ""),
            readme=build_readme(prompt_data),
            tags=prompt_data.get("tags", [])
        )
    except Exception as e:
        # O Hub recusa um commit idêntico ao anterior. O prompt já está lá na
        # versão do YAML, então o push não tem o que fazer e isso não é falha.
        if "has not changed" in str(e):
            print("O prompt já está publicado nesta versão: nada a commitar.")
            return True

        print(f"Erro ao publicar o prompt: {e}")
        return False

    print(f"Publicado com sucesso: {url}")
    return True

def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    data = load_yaml(PROMPT_FILE)
    if not data or PROMPT_KEY not in data:
        print(f"Não foi possível ler '{PROMPT_KEY}' em {PROMPT_FILE}")
        return 1

    prompt_data = data[PROMPT_KEY]

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("O prompt não passou na validação:")
        for error in errors:
            print(f" - {error}")
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    if not push_prompt_to_langsmith(f"{username}/{PROMPT_KEY}", prompt_data):
        return 1

    print("\nPróximo passo: python src/evaluate.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())