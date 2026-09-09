<div align="center">

# 🗜️ Compression & Consolidation Memory

### Turn every exchange into a dense factual statement — not a transcript.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Chat Model](https://img.shields.io/badge/Chat_Model-gpt--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/text-generation)
[![Consolidation](https://img.shields.io/badge/Consolidation-Append--only_(current)-F59E0B?style=for-the-badge)](#-compression-and-consolidation-semantics)
[![License](https://img.shields.io/badge/License-See_Root_Repo-lightgrey?style=for-the-badge)](#-license)

*Part of the [Agent Memory Optimization](https://github.com/paras160500/agent-memory-optimization) project.*

![Architecture Diagram](diagram.png)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [What the Compressor Preserves](#-what-the-compressor-preserves)
- [How It Works](#️-how-it-works)
- [Architecture](#-architecture)
- [Prompt Construction](#-prompt-construction)
- [Memory Lifecycle](#-memory-lifecycle)
- [Compression and Consolidation Semantics](#-compression-and-consolidation-semantics)
- [Characteristics](#-characteristics)
- [Advantages & Limitations](#️-advantages--limitations)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running the Demo](#-running-the-demo)
- [Usage as a Python Module](#-usage-as-a-python-module)
- [Memory API](#-memory-api)
- [Compressor API](#-compressor-api)
- [Agent API](#-agent-api)
- [Complexity & Scaling](#-complexity--scaling)
- [Choosing a Consolidation Strategy](#-choosing-a-consolidation-strategy)
- [Comparison with Other Memory Strategies](#-comparison-with-other-memory-strategies)
- [When to Use Compression and Consolidation Memory](#-when-to-use-compression-and-consolidation-memory)
- [Token & API Usage](#-token--api-usage)
- [Security & Privacy Notes](#-security--privacy-notes)
- [References](#-references)

---

## 🔍 Overview

**Compression and consolidation memory** converts each interaction into a **compact factual statement** before storing it for future context — instead of retaining complete user/assistant messages.

> 💡 **Core idea:** *Compress each conversation exchange into a dense factual memory and use the compressed memories in future prompts.*

A dedicated language-model call strips conversational filler and preserves the information that actually matters: facts, names, preferences, decisions, dates, goals, entities, and relationships.

---

## 🎯 What the Compressor Preserves

```mermaid
graph LR
    subgraph Keep["✅ Preserved"]
        direction TB
        K1["Important facts"]
        K2["Names"]
        K3["Preferences"]
        K4["Decisions"]
        K5["Dates"]
        K6["Goals"]
        K7["Important entities"]
        K8["Important relationships"]
    end
    subgraph Drop["❌ Removed"]
        direction TB
        D1["Greetings"]
        D2["Conversational filler"]
        D3["Repeated information"]
        D4["No-longer-needed explanations"]
        D5["Unnecessary wording"]
    end

    style Keep fill:#0f172a,color:#fff,stroke:#10B981
    style Drop fill:#0f172a,color:#fff,stroke:#EF4444
```

The compressor returns **only** the compressed factual statement — nothing else.

---

## ⚙️ How It Works

```mermaid
flowchart TD
    A([📩 New user message]) --> B[Retrieve all previously<br/>compressed facts]
    B --> C[Build prompt: facts +<br/>current user request]
    C --> D[Send prompt to chat model]
    D --> E[Record chat-model token usage]
    E --> F[Send user message + assistant<br/>response to MemoryCompressor]
    F --> G[Compressor strips filler,<br/>repetition, unneeded wording]
    G --> H[Store returned compressed fact]
    H --> I([📤 Return original response])

    style A fill:#4F46E5,color:#fff,stroke:#333
    style I fill:#10B981,color:#fff,stroke:#333
    style G fill:#F59E0B,color:#111,stroke:#333
```

> 📝 The next request receives the **accumulated compressed facts** instead of the complete conversation transcript.

### 🔁 Sequence of a Compressed Chat Turn

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant A as CompressionMemoryAgent
    participant M as CompressionMemory
    participant T as TokenTracker
    participant L as Chat Model
    participant C as MemoryCompressor

    U->>A: chat("What project am I working on?")
    A->>M: get_context()
    M-->>A: "### Compressed Factual Memory:<br/>-fact one<br/>-fact two"
    A->>T: estimate context tokens
    A->>L: send prompt (facts + request)
    L-->>A: response text
    A->>T: record chat usage
    A->>C: compress(user_msg, response)
    C->>L: request compressed statement (2nd call)
    L-->>C: concise factual statement
    C-->>A: compressed fact
    A->>M: add_fact(compressed_fact)
    A-->>U: original response text
```

---

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Repo["agent-memory-optimization"]
        subgraph CCM["8_compression_consolidation_memory/"]
            init["__init__.py"]
            agent["agent.py<br/><i>chat agent + compressed memory</i>"]
            demo["demo.py<br/><i>interactive CLI demo</i>"]
            compressor["compressor.py<br/><i>LLM-based compression</i>"]
            memory["memory.py<br/><i>compressed fact storage</i>"]
        end
        subgraph Common["common/"]
            llm["llm.py<br/><i>gpt-4o-mini chat model</i>"]
            tracker["token_tracker.py<br/><i>chat token usage</i>"]
        end
    end

    demo --> agent
    agent --> memory
    agent --> compressor
    agent --> llm
    agent --> tracker
    compressor --> llm
    llm -.-> OpenAI[("OpenAI API")]

    style Repo fill:#0f172a,color:#fff,stroke:#334155
    style CCM fill:#1e293b,color:#fff,stroke:#334155
    style Common fill:#1e293b,color:#fff,stroke:#334155
    style OpenAI fill:#412991,color:#fff,stroke:#333
```

---

## 🧩 Prompt Construction

```text
You are an AI assistant with compressed memory.

Previous compressed facts:

### Compressed Factual Memory:
-fact one
-fact two

Use these facts when relevant.

Current request:
<current user message>
```

> ⚠️ **Every** stored fact is included on every request. The current implementation does not rank facts, filter by relevance, or apply a token budget before sending the prompt.

`get_context()` returns `No compressed facts in memory` when the collection is empty.

---

## 🔄 Memory Lifecycle

```mermaid
flowchart TD
    A["👤 User message"] --> B["📚 Retrieve all compressed facts"]
    B --> C["📝 Build prompt"]
    C --> D["🤖 Generate assistant response"]
    D --> E["🗜️ Compress user message + response"]
    E --> F["➕ Append compressed fact"]

    style A fill:#4F46E5,color:#fff,stroke:#333
    style F fill:#10B981,color:#fff,stroke:#333
```

The stored memory is a **list of independent strings**. There is no consolidation pass that combines several facts into one summary or replaces outdated facts.

---

## ⚗️ Compression and Consolidation Semantics

Despite the module's name, the current implementation performs **compression** but only **basic accumulation** for consolidation:

```mermaid
flowchart LR
    A["✅ Compression<br/>each exchange → shorter statement"] --> B["✅ Accumulation<br/>appended to the list"]
    B --> C["❌ No merging<br/>of similar facts"]
    B --> D["❌ No deduplication"]
    B --> E["❌ No updates<br/>(old prefs never replaced)"]
    B --> F["❌ No deletion<br/>until clear()"]
    B --> G["❌ No relevance ranking<br/>(all facts sent every time)"]

    style A fill:#10B981,color:#fff,stroke:#333
    style B fill:#10B981,color:#fff,stroke:#333
    style C fill:#EF4444,color:#fff,stroke:#333
    style D fill:#EF4444,color:#fff,stroke:#333
    style E fill:#EF4444,color:#fff,stroke:#333
    style F fill:#EF4444,color:#fff,stroke:#333
    style G fill:#EF4444,color:#fff,stroke:#333
```

> 🔧 A production consolidation layer could periodically merge facts, resolve contradictions, assign confidence and timestamps, remove stale entries, and select only relevant facts for each prompt.

---

## 📊 Characteristics

| Aspect | Behavior |
| --- | --- |
| 🧩 Memory strategy | LLM-generated factual compression |
| 📦 Stored unit | One compressed fact per interaction |
| ⏱️ Compression timing | After every assistant response |
| 🔗 Consolidation behavior | Appends each fact to one collection; no dedup or merging currently implemented |
| 🔎 Retrieval | Includes **all** stored compressed facts in every subsequent prompt |
| 🤖 Chat model | `gpt-4o-mini` |
| 💾 Storage | In-memory Python list |
| ⏳ Persistence | None; facts lost when the process ends |
| 🔢 Token monitoring | Tracks the response-model context and usage metadata |
| 🎯 Best suited for | Long conversations where factual recall matters more than raw transcript detail |

---

## ⚖️ Advantages & Limitations

<table>
<tr>
<td valign="top" width="50%">

### ✅ Advantages

- **Compact memory representation** — each exchange becomes a shorter statement
- **Reduced conversational noise** — greetings, filler, repetition removed
- **Useful fact preservation** — compressor explicitly retains important info
- **Simple implementation** — no embeddings, vector DB, or graph storage needed
- **Human-readable memory** — stored facts are plain text, easy to inspect

</td>
<td valign="top" width="50%">

### ⚠️ Limitations

- **Information loss** — compression can omit details or misstate intent
- **Additional model call** — every interaction needs a separate compression request
- **Unbounded memory growth** — all facts are appended and resent forever
- **No actual consolidation** — no merging, dedup, updates, or caps
- **No relevance filtering** — every fact is sent regardless of relevance
- **No persistence** — memory disappears at process exit
- **Potential fact conflicts** — contradictions accumulate with no resolution
- **Separate usage accounting** — compression-model usage isn't tracked

</td>
</tr>
</table>

---

## 📁 Project Structure

```
8_compression_consolidation_memory/
├── __init__.py
├── agent.py        # Chat agent using compressed factual memory
├── compressor.py   # LLM-based conversation compression
├── demo.py         # Interactive command-line demonstration
└── memory.py       # Compressed fact storage and context formatting
```

The agent also uses shared modules from the repository root:

```
common/
├── llm.py           # Configures the gpt-4o-mini chat model
└── token_tracker.py # Estimates and reports chat-model token usage
```

---

## 📋 Requirements

- 🐍 Python 3.9 or later
- 🔑 An OpenAI API key
- 📦 The dependencies listed in the repository's `requirements.txt`
- 🌐 Network access for language-model requests

The implementation uses [LangChain](https://www.langchain.com/) for model integration.

---

## 🚀 Installation

**1. Clone the repository and move into its root directory:**

```bash
git clone https://github.com/paras160500/agent-memory-optimization.git
cd agent-memory-optimization
```

**2. Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

> On Windows PowerShell:
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

**3. Install the project dependencies:**

```bash
pip install -r requirements.txt
```

**4. Create a `.env` file in the repository root:**

```
OPENAI_API_KEY=your_openai_api_key_here
```

The shared LLM helper loads this value and creates a deterministic `gpt-4o-mini` chat model with `temperature=0`.

---

## ▶️ Running the Demo

Run the demo from the repository root using the package/import arrangement supported by the repository:

```bash
python -m 8_compression_consolidation_memory.demo
```

> ⚠️ **Note:** Because `8_compression_consolidation_memory` begins with a number, some Python environments do not accept it as a conventional module name. If module execution fails, rename the directory to a valid package identifier such as `compression_consolidation_memory`, update imports if needed, and run `python -m compression_consolidation_memory.demo`.

The demo starts an interactive chat session:

```text
============================================================
COMPRESSION & CONSOLIDATION MEMORY
============================================================

Memory type: Dense factual compression
LLM: gpt-4o-mini

Type 'exit' to stop.
```

Enter `exit` or `quit` to stop. After every interaction, the demo prints the newly compressed fact and the number of facts currently stored.

> ⚠️ **Note:** The current demo calls `tracker.print_current_report([])`, so its current-context report is based on an empty list rather than the actual prompt. Use `tracker.get_current_report()` or pass the constructed message list for accurate reporting.

---

## 🐍 Usage as a Python Module

After renaming the directory to a Python-valid package name if necessary:

```python
from compression_consolidation_memory.agent import CompressionMemoryAgent

agent = CompressionMemoryAgent()

print(agent.chat("My name is Alex and I am building a customer-support bot."))
print(agent.chat("What project am I working on?"))
```

The second request receives the compressed factual memories created from earlier exchanges.

**Inspect stored compressed facts:**

```python
memory = agent.get_memory()

print(f"Stored facts: {memory.count()}")

for fact in memory.get_facts():
    print(f"- {fact}")
```

**Clear all compressed facts and reset token statistics:**

```python
agent.clear_memory()
```

**Inspect token statistics:**

```python
tracker = agent.get_token_tracker()

print(tracker.get_current_report())
print(tracker.get_total_report())
```

---

## 🧠 Memory API

| Method | Description |
| --- | --- |
| `CompressionMemory()` | Creates an empty collection of compressed factual statements |
| `add_fact(compressed_fact: str)` | Appends a compressed fact to the collection |
| `get_facts() -> list[str]` | Returns every stored compressed fact in insertion order |
| `get_context() -> str` | Returns prompt-ready memory context (`"No compressed facts in memory"` if empty) |
| `count() -> int` | Returns the number of stored compressed facts |
| `clear()` | Removes every compressed fact |

---

## 🗜️ Compressor API

| Method | Description |
| --- | --- |
| `MemoryCompressor()` | Creates a compressor using the shared `gpt-4o-mini` configuration |
| `compress(user_message, ai_message) -> str` | Combines the exchange, sends it to the compression prompt, and returns the stripped response text |

> ⚠️ The method does **not** validate the output, enforce a maximum length, deduplicate facts, or preserve structured metadata.

---

## 🤖 Agent API

`CompressionMemoryAgent()` creates an agent with:

- 🤖 The shared `gpt-4o-mini` chat model
- 🗜️ A `CompressionMemory` instance
- ✍️ A `MemoryCompressor` instance
- 🔢 A `TokenTracker` configured for `gpt-4o-mini`

| Method | Description |
| --- | --- |
| `chat(user_message: str) -> str` | Retrieves compressed context, calls the chat model, compresses the completed exchange, stores the new fact, and returns the original assistant response |
| `get_memory()` | Returns the `CompressionMemory` instance |
| `get_token_tracker()` | Returns the token tracker for current and cumulative chat-model usage |
| `clear_memory()` | Clears all compressed facts and resets token statistics |

> 📝 The compressor runs **after** the response is generated — the current interaction's compressed fact is available only to later requests.

---

## 📈 Complexity & Scaling

Let `F` be the number of stored facts and `L` the average fact length. Constructing the memory context is approximately `O(F × L)`, since every fact is joined into the prompt on every request.

```mermaid
xychart-beta
    title "🗜️ Compression Memory — Context Size vs. Turns"
    x-axis "Conversation Turn" [1, 5, 10, 15, 20, 25, 30]
    y-axis "Approx. Context Tokens" 0 --> 12000
    line [200, 1200, 2400, 3600, 4800, 6000, 7200]
```

> 📝 Each interaction needs **one chat-model request plus one compression-model request**. The prompt still grows as facts accumulate — but should grow more slowly than a raw transcript if compression is effective, since filler and repetition are stripped before storage.

**For longer-running applications, consider:**

- 🔢 A maximum number of facts
- 🔁 Periodic consolidation into a smaller set
- 🔎 Relevance retrieval or embeddings
- 🧹 Fact deduplication
- 🗓️ Timestamps and confidence scores
- 🎛️ Explicit user-controlled memory deletion
- 🗄️ Persistent storage with retention policies

---

## 🎚️ Choosing a Consolidation Strategy

```mermaid
flowchart TD
    Q{What does your<br/>app need?}
    Q -->|"Simplest possible, short-lived"| S1["Append-only facts<br/>🟢 Simple, but unbounded growth"]
    Q -->|"Reduce duplication over time"| S2["Periodic merging<br/>🟡 More model calls"]
    Q -->|"Only relevant facts per query"| S3["Relevance filtering<br/>🟡 Needs ranking/embeddings"]
    Q -->|"Facts that can be updated"| S4["Structured facts<br/>(subject/value/date/confidence)<br/>🔴 More complexity"]
    Q -->|"Scale + semantic access"| S5["Hybrid compression + retrieval<br/>🔴 Needs indexing infra"]

    style S1 fill:#10B981,color:#fff
    style S2 fill:#F59E0B,color:#111
    style S3 fill:#F59E0B,color:#111
    style S4 fill:#EF4444,color:#fff
    style S5 fill:#EF4444,color:#fff
```

| Strategy | Behavior | Benefits | Trade-offs |
| --- | --- | --- | --- |
| Append-only facts *(current)* | Store every compressed fact | Simplest, preserves more source material | Memory and prompts grow indefinitely |
| Periodic merging | Combine several facts on a schedule | Reduces duplication and prompt size | Extra model calls, may lose detail |
| Relevance filtering | Include only facts related to the query | Controls prompt size | Requires ranking or embeddings |
| Structured facts | Store fields like subject, value, date, confidence | Supports updates and conflict resolution | More implementation complexity |
| Hybrid compression + retrieval | Compress facts, then retrieve relevant ones | Scales better, preserves semantic access | Requires indexing and retrieval infra |

---

## 🔬 Comparison with Other Memory Strategies

```mermaid
quadrantChart
    title Memory Strategy Trade-offs
    x-axis Low Complexity --> High Complexity
    y-axis Low Token Efficiency --> High Token Efficiency
    quadrant-1 High efficiency, high complexity
    quadrant-2 High efficiency, low complexity
    quadrant-3 Low efficiency, low complexity
    quadrant-4 Low efficiency, high complexity
    Sequential memory: [0.12, 0.15]
    Sliding-window memory: [0.22, 0.6]
    Summarization memory: [0.55, 0.78]
    Retrieval-based memory: [0.72, 0.82]
    Graph-based memory: [0.75, 0.75]
    Compression memory: [0.35, 0.55]
    Hybrid memory: [0.88, 0.9]
```

| Strategy | Stored representation | Retrieval behavior | Long-range recall | Token efficiency | Complexity |
| --- | --- | --- | --- | --- | --- |
| Sequential memory | Full transcript | Sends everything | Excellent until context limits | 🔴 Low | 🟢 Low |
| Sliding-window memory | Recent messages | Recency-based | 🔴 Low | 🟡 Medium–High | 🟢 Low |
| Summarization memory | Summary + recent messages | Generated compression | 🟡 Moderate | 🟢 High | 🟡 Medium |
| Retrieval-based memory | Embedded exchanges | Semantic similarity | 🟢 High for related text | 🟢 High | 🟡 Medium–High |
| Graph-based memory | Entities and relations | Entity and graph lookup | 🟢 High for represented relationships | 🟢 High | 🟡 Medium–High |
| **Compression memory** | Compressed factual statements | Sends all compressed facts | 🟡 Moderate–High | 🟡 Higher than raw history, but unbounded | 🟢 Low–Medium |

---

## 🎯 When to Use Compression and Consolidation Memory

**Good fit when:**
- ✅ The application mainly needs factual recall, not exact dialogue
- ✅ You want a simple, human-readable memory format
- ✅ Removing conversational filler can meaningfully shrink context size
- ✅ You're prototyping compression before adding retrieval or structured storage
- ✅ An additional compression model call is acceptable

**Consider a different strategy when:**
- ❌ The conversation needs one coherent narrative → **summarization**
- ❌ Only relevant historical facts should enter the prompt → **retrieval memory**
- ❌ Explicit relationships and multi-hop queries are central → **graph memory**

---

## 🔢 Token & API Usage

Each user interaction can involve:

1. 🤖 One chat-model request to answer the current request
2. 🗜️ One additional chat-model request to compress the completed exchange

> ⚠️ The agent's `TokenTracker` estimates and records usage for the **response-model call only**. The compressor uses the shared LLM independently and does **not** pass its usage to the agent tracker — the tracker may undercount true session cost.
>
> The demo's `print_current_report([])` call reports an empty context rather than the actual prompt. For accurate context reporting, pass the prompt message list or expose it from the agent.

For production cost accounting, track response generation and compression requests **separately**.

---

## 🔒 Security & Privacy Notes

> ⚠️ Compressed facts may contain personal, confidential, or sensitive information **even after** conversational filler is removed. Compression is **not** anonymization and should not be treated as a privacy filter.

Before production use, add:

- 🔐 Authentication and authorization
- 🗓️ Retention policies and deletion controls
- 🔒 Encryption and persistence safeguards
- 📝 Audit logging
- 🎛️ Ways for users to inspect, correct, or delete stored facts

Do not send confidential or personally identifiable information to an external model provider unless your application is designed and approved to handle it.

Calling `clear_memory()` removes the local in-memory facts but does **not** undo content already sent to an external model provider or stored under provider policies.

---

## 📄 License

Refer to the [root repository](https://github.com/paras160500/agent-memory-optimization) for the project's license and contribution guidelines.

---

## 🔗 References

| # | Resource |
| --- | --- |
| 1 | [Agent Memory Optimization repository](https://github.com/paras160500/agent-memory-optimization) |
| 2 | [`CompressionMemory` implementation](https://github.com/paras160500/agent-memory-optimization/blob/main/8_compression_consolidation_memory/memory.py) |
| 3 | [`MemoryCompressor` implementation](https://github.com/paras160500/agent-memory-optimization/blob/main/8_compression_consolidation_memory/compressor.py) |
| 4 | [`CompressionMemoryAgent` implementation](https://github.com/paras160500/agent-memory-optimization/blob/main/8_compression_consolidation_memory/agent.py) |
| 5 | [Compression and consolidation memory interactive demo](https://github.com/paras160500/agent-memory-optimization/blob/main/8_compression_consolidation_memory/demo.py) |
| 6 | [LangChain official website](https://www.langchain.com/) |
| 7 | [OpenAI text generation documentation](https://platform.openai.com/docs/guides/text-generation) |

<div align="center">

---

**⭐ Part of the [Agent Memory Optimization](https://github.com/paras160500/agent-memory-optimization) project ⭐**

</div>