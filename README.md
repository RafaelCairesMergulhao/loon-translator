# Loon Translator

Tradutor de voz bidirecional para chamadas no Windows. O Loon captura seu
microfone, traduz sua fala e envia a voz sintetizada para um microfone virtual.
Ao mesmo tempo, captura o áudio recebido pelo WASAPI loopback, traduz e reproduz
o resultado no seu fone.

## Limites importantes

- A tradução ocorre por frases. Depois que você para de falar, a voz traduzida
  começa em cerca de 1 s numa CPU comum (veja [Latência](#latência)); não
  existe tradução de voz perfeitamente instantânea.
- O Loon não instala um driver próprio. Ele usa o
  [VB-CABLE](https://vb-audio.com/Cable/), um driver assinado e mantido pela
  VB-Audio.
- A captura recebida usa o áudio inteiro do dispositivo escolhido. Para evitar
  eco, o Loon pausa essa captura enquanto reproduz a tradução recebida.
- Use fones de ouvido. Alto-falantes podem voltar ao microfone físico e causar
  eco acústico.

## Requisitos

- Windows 10 ou 11, 64 bits
- Python 3.11 ou 3.12 (Python 3.14 ainda não é suportado pela stack completa)
- VB-CABLE instalado e computador reiniciado
- Internet para tradução em nuvem e Edge TTS
- Aproximadamente 1 GB livre para ambiente, modelo e cache

## Instalação

No PowerShell, dentro desta pasta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
copy .env.example .env
python -m tools.diagnose_audio
python main.py
```

Se você já possui o `uv`, ele instala a versão correta do Python e todas as
dependências automaticamente:

```powershell
uv sync --extra dev
uv run python main.py
```

Na primeira execução, o Faster-Whisper baixa o modelo de reconhecimento
escolhido (`base` na CPU, `large-v3-turbo` com GPU NVIDIA). Esse download
ocorre uma vez e pode levar alguns minutos. Ao iniciar a tradução, o Loon
aquece os motores (status **Preparando**) para que a primeira frase já saia rápida.

Se o PowerShell bloquear a ativação do ambiente:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Configuração do VB-CABLE e da chamada

1. Baixe o VB-CABLE no site oficial, execute o instalador de 64 bits como
   administrador e reinicie o Windows.
2. No Loon, escolha seu microfone físico em **Minha voz**.
3. Escolha **CABLE Input (VB-Audio Virtual Cable)** como saída virtual.
4. No Discord, WhatsApp Desktop, jogo, Teams ou outro aplicativo, escolha
   **CABLE Output (VB-Audio Virtual Cable)** como microfone.
5. Mantenha a saída do aplicativo de chamada no seu fone.
6. No Loon, escolha o loopback que corresponde a esse mesmo fone e escolha o
   fone como saída da tradução recebida.
7. Execute **Testar fone** e depois **Iniciar tradução**.

Não marque “Escutar este dispositivo” no CABLE Output. Essa opção pode criar
realimentação. O áudio original do seu microfone não é enviado à chamada:
somente a voz traduzida chega ao CABLE Input.

## Modos de processamento

### Híbrido

Tenta os pacotes Argos e as vozes instaladas no Windows. Se não estiverem
disponíveis, usa tradução em nuvem e Edge TTS. O reconhecimento de fala
permanece local com Faster-Whisper.

### Nuvem

Usa DeepL quando `DEEPL_API_KEY` estiver definido; caso contrário, usa o backend
Google do `deep-translator`. A voz é gerada pelo Edge TTS.

### Local

É o padrão para novas instalações e não possui custo por conversa. Abra
**Pacotes de idiomas** e baixe uma vez os pares desejados. A tradução usa
inglês como interligação entre os idiomas instalados. A voz usa primeiro o
pacote de voz instalado no Windows; se escolher **Natural local (Piper)**, o
aplicativo baixa uma voz neural (~60 MB) e passa a falar offline com qualidade
bem melhor que a voz clássica do Windows. Também é possível apontar um modelo
com `PIPER_MODEL`.

Os 12 idiomas de lançamento são português, inglês, espanhol, francês, alemão,
italiano, hindi, mandarim, japonês, coreano, russo e árabe. O mecanismo não
envia conteúdo à nuvem silenciosamente no modo Local.

### IA (LLM)

Traduz com um modelo de linguagem que recebe as últimas falas da conversa e o
registro escolhido (casual, formal, viagem). Isso resolve gírias, pronomes e
ambiguidades que um tradutor frase a frase erra. Funciona com qualquer servidor
compatível com a API da OpenAI; configure no `.env`:

| Provedor     | Onde roda | Chave | Modelo padrão             |
|--------------|-----------|-------|---------------------------|
| `ollama`     | local     | não   | `qwen2.5:3b`              |
| `lmstudio`   | local     | não   | modelo carregado          |
| `groq`       | nuvem     | sim   | `qwen/qwen3.8-27b`        |
| `openai`     | nuvem     | sim   | `gpt-4o-mini`             |
| `gemini`     | nuvem     | sim   | `gemini-2.0-flash`        |
| `openrouter` | nuvem     | sim   | configurável              |

Para rodar 100% local: instale o [Ollama](https://ollama.com), execute
`ollama pull qwen2.5:3b` e deixe `LLM_PROVIDER=ollama`. Se a LLM falhar ou
passar de `LLM_TIMEOUT` segundos, o Loon cai automaticamente para
Argos → DeepL → Google e deixa a LLM em pausa por 30 s (disjuntor), para que uma
falha não atrase as frases seguintes.

## Latência

Medido nesta máquina (CPU, sem GPU) com a mesma frase em português de 9 s,
do fim da fala até o início da voz traduzida em inglês:

| Versão | Início da voz | Transcrição                          |
|--------|---------------|--------------------------------------|
| 0.3.0  | 7,1 – 10,7 s  | errada (`distil-large-v3` só entende inglês) |
| 0.4.0  | 0,8 – 1,0 s   | correta (`base`, int8, 8 threads)    |

De onde veio o ganho:

- **Reconhecimento**: o seletor de fala (Automático, Rápido, Equilibrado,
  Preciso) escolhe o modelo pelo hardware. O Whisper sempre processa uma janela
  de 30 s, então o custo por frase é quase fixo: `base` ≈ 0,65 s, `small` ≈ 2 s.
- **Voz do Windows**: um processo SAPI persistente substitui um PowerShell novo
  por frase (385 ms → 35 ms).
- **Fala por sentenças**: a primeira sentença é tocada enquanto as seguintes são
  sintetizadas (produtor–consumidor com filas por sentido).
- **Silêncio cortado**: 0,1 s no início e 0,66 s no fim de cada frase da SAPI.
- **Ajuste de timbre**: energia por janela com somas de prefixo, O(n) em vez de
  O(n·janela) (455 ms → 45 ms).
- **Velocidade adaptativa**: com frases na fila, a fala acelera 8% por frase
  pendente (até 1,5×) para não acumular atraso.
- **Pré-roll de 200 ms** no detector de frases: a primeira sílaba não é cortada.
- **Cache LRU** de traduções (256 entradas) e keep-alive HTTP.

Para medir no seu computador:

```powershell
python -m tools.benchmark_latency --models base small --legacy
```

O histórico da interface mostra cada etapa (`STT · MT · TTS`) e os percentis
p50/p95 da sessão.

## Estruturas de dados e algoritmos

`domain/sorting.py` implementa os oito métodos clássicos de ordenação, com
contagem de comparações C(n) e movimentações M(n), e o aplicativo usa cada
um onde as suas propriedades importam:

| Método          | Onde é usado                                                        | Por quê |
|-----------------|---------------------------------------------------------------------|---------|
| Inserção direta | fila de provedores de tradução por saúde                            | n < 10, quase ordenada, estável: O(n) |
| Shellsort       | lista de dispositivos de áudio                                      | dezenas de itens, sem recursão nem memória extra |
| Mergesort       | seleção de trechos de gírias sobrepostos                            | estável: empate mantém a ordem do texto |
| Heap binário    | top-k arquivos mais recentes do cache de voz                        | O(n log k) em vez de ordenar tudo |
| Quicksort (Hoare) | Quickselect para p50/p95 de latência                              | k-ésimo elemento em O(n) médio |
| Seleção, Bolha, Shakersort | comparação didática                                      | — |

Outros conceitos aplicados: fila circular (histórico de latência), cache LRU,
disjuntor (circuit breaker) com média móvel exponencial da latência dos
provedores, pipeline produtor–consumidor com filas limitadas, soma de prefixos
e buffer de pré-roll.

Benchmark dos métodos (tempo, C(n), M(n) em dados aleatórios, ordenados,
invertidos e quase ordenados):

```powershell
python -m tools.benchmark_sorting --sizes 100 1000 2000
```

Com n = 1000, os valores medidos confirmam a teoria: Seleção direta faz
sempre n(n−1)/2 = 499.500 comparações; Bolha em vetor invertido faz
3·n(n−1)/2 = 1.498.500 movimentações; Inserção em vetor ordenado faz n−1 = 999
comparações; Quicksort, Mergesort, Heapsort e Shellsort ficam entre 6 mil e
17 mil comparações.

## Privacidade

- O Faster-Whisper processa o áudio localmente.
- Nos modos Híbrido/Nuvem, o texto pode ser enviado ao provedor de tradução e o
  texto traduzido ao Edge TTS.
- No modo IA com provedor na nuvem, o texto e as últimas falas da conversa
  (até 6) vão para o provedor escolhido. Com Ollama ou LM Studio, nada sai do
  computador.
- O histórico não é salvo por padrão. Quando habilitado, fica em
  `%APPDATA%\LoonTranslator\loon_translator.db`.
- Preferências ficam em `%APPDATA%\LoonTranslator\settings.json`.
- Áudios sintetizados ficam em um cache local limitado aos 100 arquivos mais
  recentes em `%APPDATA%\LoonTranslator\cache`.
- Logs técnicos ficam em `%APPDATA%\LoonTranslator\loon.log` e não incluem
  transcrições por padrão.
- O cliente Firebase antigo não participa do fluxo e nenhuma conversa é
  enviada ao Firebase.
- A interface pede autorização antes do primeiro uso de serviços em nuvem.
  Use **Apagar histórico e cache** para remover conversas salvas e vozes
  sintetizadas.

## Diagnóstico

```powershell
python -m tools.diagnose_audio
```

O comando deve indicar `VB-CABLE: OK` e `WASAPI loopback: OK`.

Se não houver loopback, confirme que o pacote `PyAudioWPatch` foi instalado,
que existe um dispositivo de reprodução ativo e que o Windows usa WASAPI. Se o
VB-CABLE não aparecer, reinstale-o como administrador e reinicie o computador.

Se a voz estiver cortada, aproxime o microfone e reduza ruído ambiente. Os
parâmetros avançados `phrase_silence_ms` e `max_phrase_seconds` são gravados em
`settings.json`.

## Testes

```powershell
pip install -e ".[dev]"
pytest
```

Checklist manual:

1. Testar o fone pela interface.
2. Confirmar no gravador do Windows que CABLE Output recebe a voz traduzida.
3. Fazer uma chamada de teste e verificar os dois sentidos.
4. Confirmar que a tradução recebida não é retranscrita em ciclo.
5. Desconectar/reconectar um dispositivo e usar **Atualizar dispositivos**.
6. Testar a parada durante fala, durante tradução e durante reprodução.
7. Desligar a rede para confirmar o erro explícito no modo Nuvem e o
   funcionamento do STT local.

## Empacotamento

Para gerar o instalador que pode ser enviado a outras pessoas:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

O script roda os testes, gera o ícone (`tools/make_icon.py`), empacota com o
PyInstaller em `%LOCALAPPDATA%\LoonBuild` (fora do OneDrive, que trava a pasta
de build) e compila `installer\LoonTranslator.iss` com o Inno Setup (instalado
pelo winget se faltar). O resultado é `release\LoonTranslator-0.5.0-Setup.exe`
com o SHA-256 ao lado.

O instalador não pede administrador, cria atalhos, mostra o guia
`installer\LEIA-ME.txt` e oferece o site do VB-CABLE quando o driver não está
instalado. O `.env` do desenvolvedor nunca entra no pacote (o script aborta se
encontrar um): no app instalado, as chaves ficam em
`%APPDATA%\LoonTranslator\.env`, e o modo IA pede a chave do Groq na primeira
vez.

Para gerar só a pasta do executável, sem instalador:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

O resultado fica em `dist\LoonTranslator`. Leia
`docs\COMMERCIAL_RELEASE.md` antes de enviar à Steam. O aplicativo não embute
nem instala silenciosamente o VB-CABLE; o driver deve vir do fornecedor
oficial.
