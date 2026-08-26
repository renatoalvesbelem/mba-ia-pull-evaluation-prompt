"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import re
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"

def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def prompt():
    """Prompt otimizado carregado do YAML."""
    data = load_prompts(PROMPT_FILE)
    assert PROMPT_KEY in data, f"Chave '{PROMPT_KEY}' não encontrada em {PROMPT_FILE.name}"
    return data[PROMPT_KEY]

@pytest.fixture(scope="module")
def full_text(prompt):
    """System e user prompt concatenados, para buscas de conteúdo."""
    return f"{prompt.get('system_prompt', '')}\n{prompt.get('user_prompt', '')}"

class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt, "Campo 'system_prompt' não existe no YAML"

        system_prompt = prompt["system_prompt"]
        assert isinstance(system_prompt, str), "'system_prompt' deve ser texto"
        assert system_prompt.strip(), "'system_prompt' está vazio"

    def test_prompt_has_role_definition(self, prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt["system_prompt"]

        match = re.search(r"Você é (?:um|uma)\s+([^\n.,]+)", system_prompt, re.IGNORECASE)
        assert match, "O system_prompt não define uma persona no padrão 'Você é um(a) ...'"

        role = match.group(1).strip()
        assert len(role) > 3, f"Persona definida é genérica demais: '{role}'"

    def test_prompt_mentions_format(self, full_text):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        assert "Markdown" in full_text, "O prompt não exige formato Markdown"

        template = re.search(
            r"Como\s+<?persona>?.*eu quero.*para que",
            full_text,
            re.IGNORECASE | re.DOTALL
        )
        assert template, "O prompt não exige o template 'Como ..., eu quero ..., para que ...'"

        assert "Critérios de Aceitação" in full_text, \
            "O prompt não exige a seção de Critérios de Aceitação"

    def test_prompt_has_few_shot_examples(self, prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt["system_prompt"]

        examples = re.findall(r"###\s*Exemplo\s+\d+", system_prompt)
        assert len(examples) >= 2, \
            f"Few-shot exige pelo menos 2 exemplos, encontrados: {len(examples)}"

        assert system_prompt.count("Relato:") >= len(examples), \
            "Nem todos os exemplos apresentam o relato de entrada"
        assert system_prompt.count("Resposta:") >= len(examples), \
            "Nem todos os exemplos apresentam a resposta esperada"

    def test_prompt_no_todos(self, prompt):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        for field, value in prompt.items():
            if not isinstance(value, str):
                continue

            for marker in ("[TODO]", "TODO", "FIXME", "XXX", "<preencher>"):
                assert marker not in value, \
                    f"Marcação '{marker}' encontrada no campo '{field}'"

    def test_minimum_techniques(self, prompt):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt.get("techniques_applied", [])

        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, \
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        assert all(str(t).strip() for t in techniques), \
            "Há técnica vazia na lista 'techniques_applied'"

        assert any("few" in str(t).lower() for t in techniques), \
            f"Few-shot Learning é obrigatório e não está listado: {techniques}"

    def test_prompt_structure_is_valid(self, prompt):
        """Valida a estrutura do prompt com o utilitário do projeto."""
        is_valid, errors = validate_prompt_structure(prompt)
        assert is_valid, f"Estrutura inválida: {errors}"

    def test_bug_report_variable_only_in_user_prompt(self, prompt):
        """O relato deve entrar só pelo user_prompt (na v1 estava duplicado no system)."""
        assert "{bug_report}" in prompt.get("user_prompt", ""), \
            "O user_prompt não recebe a variável {bug_report}"
        assert "{bug_report}" not in prompt.get("system_prompt", ""), \
            "A variável {bug_report} não deve ser repetida no system_prompt"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])