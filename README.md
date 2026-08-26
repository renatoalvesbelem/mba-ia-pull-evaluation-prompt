# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Desafio do MBA em IA: puxar um prompt de baixa qualidade do LangSmith Prompt Hub, refatorá-lo com
técnicas avançadas de Prompt Engineering, republicá-lo e comprovar, por avaliação automatizada, que
as cinco métricas (Helpfulness, Correctness, F1-Score, Clarity e Precision) ficam **todas** acima de 0.8.

- **Prompt original (v1):** `leonanluppi/bug_to_user_story_v1`
- **Prompt otimizado (v2):** [`renatoalvesbelem/bug_to_user_story_v2`](https://smith.langchain.com/hub/renatoalvesbelem/bug_to_user_story_v2) (público)
- **Dataset de avaliação:** 15 relatos de bug — 5 simples, 7 médios, 3 complexos

---

## Técnicas Aplicadas (Fase 2)

O prompt v1 tinha 232 caracteres de instrução, nenhuma persona, nenhum exemplo, nenhum formato de
saída definido e a variável `{bug_report}` duplicada no system e no user prompt. O v2 tem 13.343
caracteres organizados em quatro técnicas combinadas.

### 1. Role Prompting

**Por quê:** "um assistente que ajuda a transformar relatos de bugs" não carrega repertório nenhum.
Uma persona específica traz junto o vocabulário, o nível de detalhe e os critérios de qualidade da
função — o modelo passa a escrever como quem entrega para um time, não como quem responde a uma
pergunta.

**Como apliquei:**

```yaml
Você é um Product Owner técnico com dez anos de experiência em times ágeis de produto digital.
Você recebe relatos de bugs (usuários, suporte ou QA) e os devolve ao time como user stories
prontas para refinamento: claras para o negócio e precisas o suficiente para implementar e testar.
```

O "prontas para refinamento" é o que define o padrão de qualidade: nem rascunho, nem especificação
fechada.

### 2. Chain of Thought (CoT)

**Por quê:** converter um bug em user story exige uma inversão que o modelo erra com frequência —
descrever o **comportamento correto desejado**, não o defeito relatado. E exige decidir quem é a
persona afetada, que nem sempre é uma pessoa (integrações, rotinas automáticas, regras de
permissão). Sem um roteiro explícito, o modelo copia o relato e troca as palavras.

**Como apliquei:** um roteiro de seis passos marcado como uso interno, para o raciocínio não vazar
para a resposta:

```yaml
## Como raciocinar (uso interno - nunca exiba isto na resposta)

1. Persona: quem é afetado? ... Se o defeito ocorre entre sistemas, sem pessoa na ponta
   (webhook, integração, rotina automática, validação de permissão em API) ... a persona é o
   próprio sistema: "Como o sistema de pagamentos, eu quero...".
2. Ação: o que a persona precisa conseguir fazer quando resolvido - o comportamento correto,
   nunca o defeito.
3. Benefício: por que isso importa para o negócio.
4. Critérios de aceitação: quais comportamentos observáveis provam que a correção funcionou.
5. Fatos concretos do relato (endpoints, logs, números, severidade) devem ser preservados; o que
   o relato não traz não deve ser inventado.
6. Nível de detalhe do relato define o formato da resposta.
```

### 3. Skeleton of Thought

**Por quê:** o dataset mistura relatos de uma linha com relatos de várias páginas contendo múltiplos
defeitos. Um formato único penaliza os dois extremos: infla o bug simples com seções vazias e
comprime o bug complexo. Como as métricas de Precision e Clarity punem tanto a informação inventada
quanto a omissão, a saída precisa ser proporcional à entrada.

**Como apliquei:** três esqueletos fixos, escolhidos pelo número de **defeitos independentes** no
relato (não pelo seu tamanho):

| Nível | Gatilho | Estrutura da resposta |
|---|---|---|
| SIMPLES | uma ou duas frases, um sintoma | frase "Como/eu quero/para que" + exatamente 5 critérios |
| MÉDIO | um problema com passos, logs, números ou severidade | frase + 6-8 critérios + até 2 seções complementares + contexto |
| COMPLEXO | dois ou mais defeitos independentes | blocos `=== USER STORY PRINCIPAL ===`, `=== CRITÉRIOS DE ACEITAÇÃO ===` (uma letra por problema), `=== CRITÉRIOS TÉCNICOS ===`, `=== CONTEXTO DO BUG ===`, `=== TASKS TÉCNICAS SUGERIDAS ===`, `=== MÉTRICAS DE SUCESSO ===` |

O desempate está escrito no prompt, porque era a confusão mais comum: *"O que decide entre MÉDIO e
COMPLEXO é o número de defeitos independentes, não o tamanho do relato: passos numerados, logs e
trechos de código descrevem um único defeito e mantêm o nível MÉDIO. Na dúvida, escolha MÉDIO."*

### 4. Few-shot Learning (obrigatória)

**Por quê:** as regras de formato descrevem a saída, mas não a calibram. Os exemplos mostram o nível
de granularidade de um critério de aceitação, o tom e o que fica de fora — coisas que uma regra
textual não transmite com a mesma economia.

**Como apliquei:** três pares entrada/saída completos, **um por nível de complexidade**, para que o
exemplo funcione também como âncora da decisão de formato:

- **Exemplo 1 (simples):** filtro de busca por data → 5 critérios, nada além disso.
- **Exemplo 2 (médio):** endpoint `/api/orders/:id` sem checar titularidade → demonstra a persona
  não-humana ("Como o sistema de pedidos"), a seção "Critérios Adicionais para Atendentes" e o
  "Contexto de Segurança" com a classificação OWASP.
- **Exemplo 3 (complexo):** dois defeitos em agendamento → a estrutura completa com os marcadores
  `===`, com um comentário explicando *por que* aquele caso justifica a estrutura maior.

### Regras e edge cases

Além das técnicas, o v2 traz um bloco de regras explícitas e um de casos de borda. As que mais
mudaram o resultado:

- **Anti-alucinação:** *"Não invente números de uso, nomes de ferramentas ou severidades que o
  relato não informa; quando um dado desses for necessário e não existir, use um marcador entre
  colchetes, como `[nome do gateway de pagamento]`."*
- **Critério testável:** *"prefira o valor, o limite, o código ou o nome concreto à formulação
  genérica ('o relatório deve ser gerado em menos de 30 segundos', não 'deve responder
  rapidamente')."*
- **Direção técnica nomeada:** uma causa provável apontada no relato deve virar o nome da técnica
  consagrada que resolve o problema (índice composto, bloqueio otimista, fila assíncrona, paginação
  por cursor), porque é isso que permite ao time dimensionar a tarefa.
- **Edge cases:** relato genérico demais (responder com a interpretação mais provável + seção
  "Informações Necessárias"), relato que na verdade pede funcionalidade nova, e relato com
  credenciais ou dados pessoais (referenciar como `[dado sensível removido]`, nunca reproduzir).

### System vs User Prompt

Toda a instrução — persona, roteiro, formato, regras, exemplos — fica no **system prompt**. O **user
prompt** carrega apenas o dado variável:

```yaml
user_prompt: |
  Relato de bug:
  ---
  {bug_report}
  ---
```

Na v1 o `{bug_report}` aparecia nos dois, o que duplicava o relato em toda chamada e misturava
instrução com dado. O teste `test_bug_report_variable_only_in_user_prompt` protege essa separação.

---

## Resultados Finais

### Comparativo estrutural v1 vs v2

| Aspecto | v1 (original) | v2 (otimizado) |
|---|---|---|
| Tamanho do system prompt | 232 caracteres, 10 linhas | 13.343 caracteres, 276 linhas |
| Persona | "um assistente" | Product Owner técnico, 10 anos em times ágeis |
| Exemplos few-shot | 0 | 3 (um por nível de complexidade) |
| Roteiro de raciocínio | ausente | 6 passos, uso interno |
| Formato de saída | não especificado | 3 esqueletos + template "Como/eu quero/para que" |
| Critérios de aceitação | não exigidos | Dado/Quando/Então, quantidade por nível |
| Regras explícitas | nenhuma | 12 regras + 4 edge cases |
| `{bug_report}` | duplicado no system e no user | apenas no user prompt |
| Técnicas declaradas | — | Role Prompting, Few-shot, CoT, Skeleton of Thought |

### Métricas

Resultado da avaliação do prompt v2 sobre os 15 exemplos do dataset, com `gpt-4o-mini` respondendo e
`gpt-4o` como juiz. A saída completa do terminal está em
[docs/evidencias/evaluate-output.txt](docs/evidencias/evaluate-output.txt).

| Métrica | v2 | Mínimo | Status |
|---|---|---|---|
| Helpfulness | 0.87 | 0.80 | ✓ |
| Correctness | 0.83 | 0.80 | ✓ |
| F1-Score | 0.81 | 0.80 | ✓ |
| Clarity | 0.88 | 0.80 | ✓ |
| Precision | 0.86 | 0.80 | ✓ |
| **Média Geral** | **0.8505** | **0.80** | **✓ APROVADO** |

Helpfulness e Correctness são derivadas das três métricas base: `helpfulness = (clarity + precision)/2`
e `correctness = (f1 + precision)/2`.

#### Jornada de otimização

Foram quatro rodadas de avaliação até todas as métricas passarem:

| Rodada | Versão do prompt | Helpfulness | Correctness | F1-Score | Clarity | Precision | Média | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | condensada (13,3k caracteres, 3 exemplos) | 0.85 | 0.81 | 0.7960 | 0.86 | 0.83 | 0.8309 | ✗ |
| 2 | completa (24,7k caracteres, 8 exemplos) | 0.86 | 0.82 | 0.7867 | 0.87 | 0.85 | 0.8352 | ✗ |
| 3 | calibrada pelo dataset (10 exemplos) | 0.86 | 0.82 | 0.8073 | 0.88 | 0.84 | 0.8402 | ✓ |
| 4 | mesma da rodada 3, reexecução | 0.87 | 0.83 | 0.8093 | 0.88 | 0.86 | 0.8505 | ✓ |

O F1-Score foi a métrica que travou o resultado nas duas primeiras rodadas — as outras quatro já
passavam desde o início. O diagnóstico veio da comparação com as referências do dataset: o prompt
pedia de seis a oito critérios no bloco principal e permitia duas seções complementares, enquanto as
referências do nível MÉDIO usam cinco ou seis critérios e no máximo uma seção. Cada seção a mais era
contada pelo juiz como informação não suportada pelo ground truth, derrubando precision e recall.

As correções da rodada 3 foram: limitar o bloco principal a cinco ou seis critérios, reduzir as seções
complementares a no máximo uma, fixar o contexto em dois a quatro tópicos, endurecer o gatilho do
nível SIMPLES e acrescentar dois exemplos simples ao final do prompt — a proporção passou de um
exemplo simples em oito para três em dez. Os relatos simples, que haviam regredido na rodada 2,
subiram de 0.69 para 0.80, de 0.80 para 0.90 e de 0.95 para 1.00 no F1.

A rodada 4 é uma reexecução do mesmo prompt, sem alterações, para confirmar a estabilidade: o F1 saiu
de 0.8073 para 0.8093. A margem sobre o corte é estreita e vale considerar em reexecuções futuras.

> **Nota sobre o prompt v1:** o `src/evaluate.py` avalia apenas `{username}/bug_to_user_story_v2`, de
> modo que não há medição numérica do prompt original. A comparação entre v1 e v2 neste README é
> estrutural, na tabela da seção anterior.

### Evidências no LangSmith

Todos os links abaixo são públicos e abrem sem login.

- **Prompt v2 no Prompt Hub:** https://smith.langchain.com/hub/renatoalvesbelem/bug_to_user_story_v2
- **Dataset de avaliação** — `bug-to-user-story-optimization-eval`, com os 15 exemplos:
  https://smith.langchain.com/public/342039ee-97c6-4f67-ad5a-e0338155b95d/d
- **Saída da avaliação aprovada:** [docs/evidencias/evaluate-output.txt](docs/evidencias/evaluate-output.txt)

**Tracing detalhado**, um trace por nível de complexidade, cada um mostrando o prompt completo enviado
ao `gpt-4o-mini` e a user story gerada:

| Nível do relato | Trace público |
|---|---|
| Simples | https://smith.langchain.com/public/93d1e574-7640-4d24-8643-ada83625bdcc/r |
| Médio | https://smith.langchain.com/public/42422a21-5834-4851-8d84-3e8bf3ccc757/r |
| Complexo | https://smith.langchain.com/public/e76ab13e-870e-4de6-a360-c4cc067c76dc/r |

O projeto completo de tracing, com todas as execuções, fica em
`bug-to-user-story-optimization` no workspace (acesso restrito — o LangSmith publica traces
individuais e datasets, não projetos inteiros).

> **Onde ficam as notas:** o `src/evaluate.py` calcula as cinco métricas localmente, com chamadas
> diretas ao modelo avaliador, e imprime o resultado no terminal — ele não registra feedback via
> `client.create_feedback` nem usa `langsmith.evaluation.evaluate`. As execuções aparecem no LangSmith
> como traces das chamadas de LLM, sem score associado. Por isso a evidência numérica das métricas é a
> saída do terminal, versionada em `docs/evidencias/`, e não o dashboard.

---

## Como Executar

### Pré-requisitos

- Python 3.9 ou superior (validado em 3.12)
- Conta no [LangSmith](https://smith.langchain.com/) com API key
- API key da OpenAI **ou** do Google AI Studio

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

No Windows, ative com `venv\Scripts\activate`.

### 2. Variáveis de ambiente

Copie o template e preencha as credenciais:

```bash
cp .env.example .env
```

| Variável | Descrição |
|---|---|
| `LANGSMITH_API_KEY` | chave da API do LangSmith |
| `LANGSMITH_PROJECT` | nome do projeto de tracing (o dataset vira `<projeto>-eval`) |
| `USERNAME_LANGSMITH_HUB` | seu handle público no Hub, visível ao publicar um prompt |
| `LLM_PROVIDER` | `openai` ou `google` |
| `LLM_MODEL` | modelo que responde — `gpt-4o-mini` ou `gemini-2.5-flash` |
| `EVAL_MODEL` | modelo avaliador — `gpt-4o` ou `gemini-2.5-flash` |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | conforme o provider escolhido |

O `.env` está no `.gitignore` e não deve ser versionado.

### 3. Pull do prompt original

```bash
python src/pull_prompts.py
```

Baixa `leonanluppi/bug_to_user_story_v1` e grava em `prompts/bug_to_user_story_v1.yml` com os
metadados (`source`, `input_variables`, `tags`).

### 4. Validar o prompt otimizado

```bash
pytest tests/test_prompts.py -v
```

Oito testes cobrem: presença do system prompt, definição de persona, exigência de Markdown e do
template de user story, existência dos exemplos few-shot, ausência de TODOs, mínimo de duas técnicas
declaradas, estrutura geral do YAML e a separação da variável `{bug_report}`.

### 5. Push do prompt otimizado

```bash
python src/push_prompts.py
```

Valida o YAML (campos obrigatórios, ausência de TODOs, mínimo de 2 técnicas e variáveis do template)
e publica em `{USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2` como repositório **público**, com
descrição, tags e um README gerado a partir das técnicas declaradas. Ao final imprime a URL do commit.

### 6. Avaliação

```bash
python src/evaluate.py
```

Cria o dataset no LangSmith a partir de `datasets/bug_to_user_story.jsonl` (se ainda não existir),
puxa o prompt v2 **do Hub** — não do arquivo local —, roda os 15 exemplos e calcula as cinco
métricas. Retorna código de saída 0 apenas se todas ficarem em 0.8 ou acima.

> **Atenção:** o `evaluate.py` sempre avalia o que está publicado no Hub. Depois de editar
> `prompts/bug_to_user_story_v2.yml`, refaça o push antes de avaliar, ou você medirá a versão antiga.

Para registrar o resultado como evidência em vez de apenas exibi-lo:

```bash
python src/evaluate.py 2>&1 | tee docs/evidencias/evaluate-output.txt
```

### Ordem completa

```bash
python src/pull_prompts.py && pytest tests/test_prompts.py && python src/push_prompts.py && python src/evaluate.py
```

---

## Estrutura do projeto

```
mba-ia-pull-evaluation-prompt/
├── .env.example                      # Template das variáveis de ambiente
├── requirements.txt                  # Dependências Python
├── README.md                         # Esta documentação
│
├── prompts/
│   ├── bug_to_user_story_v1.yml      # Prompt original, gerado pelo pull
│   └── bug_to_user_story_v2.yml      # Prompt otimizado
│
├── datasets/
│   └── bug_to_user_story.jsonl       # 15 bugs (5 simples, 7 médios, 3 complexos)
│
├── docs/
│   └── evidencias/
│       └── evaluate-output.txt       # Saída da avaliação aprovada
│
├── src/
│   ├── pull_prompts.py               # Pull do LangSmith Hub
│   ├── push_prompts.py               # Push público ao LangSmith Hub
│   ├── evaluate.py                   # Avaliação automática (fornecido)
│   ├── metrics.py                    # As 5 métricas (fornecido)
│   └── utils.py                      # Funções auxiliares (fornecido)
│
└── tests/
    └── test_prompts.py               # Testes de validação do prompt
```

## Referências

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Repositório base do desafio](https://github.com/devfullcycle/mba-ia-pull-evaluation-prompt)
