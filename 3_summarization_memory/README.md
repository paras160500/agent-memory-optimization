<div align="center">

# 🧾 Summarization Memory

### Compress older history into a summary. Keep the newest messages verbatim.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/Model-gpt--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/text-generation)
[![Recent Messages](https://img.shields.io/badge/Default_Recent-4_messages-F59E0B?style=for-the-badge)](#-choosing-max_recent_messages)
[![License](https://img.shields.io/badge/License-See_Root_Repo-lightgrey?style=for-the-badge)](#-license)

*Part of the [Agent Memory Optimization](https://github.com/paras160500/agent-memory-optimization) project.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [How It Works](#️-how-it-works)
- [Architecture](#-architecture)
- [Context Construction](#-context-construction)
- [Characteristics](#-characteristics)
- [Advantages & Limitations](#️-advantages--limitations)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running the Demo](#-running-the-demo)
- [Usage as a Python Module](#-usage-as-a-python-module)
- [Memory API](#-memory-api)
- [Summarizer API](#-summarizer-api)
- [Agent API](#-agent-api)
- [Message-Level Retention](#-message-level-retention)
- [Complexity & Scaling](#-complexity--scaling)
- [Choosing `max_recent_messages`](#-choosing-max_recent_messages)
- [Comparison with Other Memory Strategies](#-comparison-with-other-memory-strategies)
- [When to Use Summarization Memory](#-when-to-use-summarization-memory)
- [Token Tracking](#-token-tracking)
- [Security & Privacy Notes](#-security--privacy-notes)
- [References](#-references)

---

## 🔍 Overview

**Summarization memory** is designed for conversations that are too long to retain in full. Instead of sending the entire history to the language model, the agent periodically **summarizes messages that fall outside a recent-message window**. The resulting summary becomes long-term conversational memory, while the newest messages stay available in their original form.

> 💡 **Core idea:** *Compress older conversation history into a summary and keep the most recent messages unchanged.*

The `SummarizationMemoryAgent` maintains **two types of context**:

| Context type | Description |
| --- | --- |
| 🗒️ **Long-term summary** | A model-generated summary of older conversation messages |
| 💬 **Recent messages** | The newest messages, retained in their original LangChain format |

---

## ⚙️ How It Works

```mermaid
flowchart TD
    A([📩 New user message]) --> B[Add to memory]
    B --> C{Messages older than<br/>recent-message limit?}
    C -->|No| G[Build context:<br/>summary + recent messages]
    C -->|Yes| D[Send old messages + existing<br/>summary to ConversationSummarizer]
    D --> E[Replace old summary with<br/>newly generated summary]
    E --> F[Remove summarized messages]
    F --> G
    G --> H[Track estimated context size]
    H --> I[Send context to LLM]
    I --> J[Record API usage]
    J --> K[Append assistant response to memory]
    K --> L([📤 Return response])

    style A fill:#4F46E5,color:#fff,stroke:#333
    style L fill:#10B981,color:#fff,stroke:#333
    style D fill:#F59E0B,color:#111,stroke:#333
    style E fill:#F59E0B,color:#111,stroke:#333
```

### 🔁 Sequence of a Summarized Chat Turn

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant A as SummarizationMemoryAgent
    participant M as SummarizationMemory
    participant S as ConversationSummarizer
    participant T as TokenTracker
    participant L as LLM (gpt-4o-mini)

    U->>A: chat("What project am I working on?")
    A->>M: add_user_message()
    A->>M: get_old_messages()
    alt Old messages exist
        M-->>A: [messages beyond window]
        A->>S: summarize(existing_summary, old_messages)
        S->>L: request updated summary
        L-->>S: new summary text
        S-->>A: updated summary
        A->>M: update_summary()
        A->>M: remove_old_messages()
    end
    A->>M: get_summary() + get_recent_messages()
    M-->>A: SystemMessage(summary) + recent messages
    A->>T: estimate context tokens
    A->>L: send context
    L-->>A: response text
    A->>T: record token usage
    A->>M: add_ai_message()
    A-->>U: response text
```

---

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Repo["agent-memory-optimization"]
        subgraph SUM["3_summarization_memory/"]
            init["__init__.py"]
            agent["agent.py<br/><i>combines summary + recent</i>"]
            demo["demo.py<br/><i>interactive CLI demo</i>"]
            memory["memory.py<br/><i>summary + recent-message state</i>"]
            summarizer["summarizer.py<br/><i>LLM-based summarization</i>"]
            readme["README.md"]
        end
        subgraph Common["common/"]
            llm["llm.py<br/><i>LangChain OpenAI client</i>"]
            tracker["token_tracker.py<br/><i>token usage estimator</i>"]
        end
    end

    demo --> agent
    agent --> memory
    agent --> summarizer
    agent --> tracker
    summarizer --> llm
    agent --> llm
    llm -.-> OpenAI[("OpenAI API")]

    style Repo fill:#0f172a,color:#fff,stroke:#334155
    style SUM fill:#1e293b,color:#fff,stroke:#334155
    style Common fill:#1e293b,color:#fff,stroke:#334155
    style OpenAI fill:#412991,color:#fff,stroke:#333
```

---

## 🧩 Context Construction

When a summary exists, the agent sends a `SystemMessage` followed by the recent raw messages:

```mermaid
graph TD
    subgraph Context["Model Context"]
        direction TB
        SysMsg["🗒️ SystemMessage<br/>'Here is the user's long-term memory:<br/>&lt;generated summary&gt;'"]
        R1["💬 Recent message 1"]
        R2["💬 Recent message 2"]
        R3["💬 Recent message 3"]
        Rn["💬 ... up to max_recent_messages"]
        SysMsg --> R1 --> R2 --> R3 --> Rn
    end

    style SysMsg fill:#F59E0B,color:#111,stroke:#333
    style Context fill:#0f172a,color:#fff,stroke:#334155
```

When no summary exists yet, the context contains **only** the recent messages. The summary is generated from older messages *before* they're removed — so the recent-message list stays bounded while the summary carries forward selected historical information.

The summarizer is explicitly instructed to preserve:

```mermaid
mindmap
  root((Summary<br/>Contents))
    Identity
    Preferences
    Important facts
    Projects
    Goals
    Decisions
    Important context
```

---

## 📊 Characteristics

| Aspect | Behavior |
| --- | --- |
| 🧩 Memory strategy | Generated summary plus recent raw messages |
| 🗒️ Summary content | Identity, preferences, facts, projects, goals, decisions, important context |
| 🔢 Recent-message default | `4` messages (memory class and agent constructor) |
| ⚠️ Demo configuration | `3` recent messages — see [note below](#demo-configuration-note) |
| 📦 Storage format | Summary string plus LangChain `HumanMessage`/`AIMessage` objects |
| 🚨 Summary trigger | Whenever stored messages exceed the recent-message limit |
| 💾 Persistence | In-memory only; lost when the process ends |
| 🔢 Token monitoring | Estimates active context tokens and records API usage |
| 🤖 Default model | `gpt-4o-mini` |
| 🎯 Best suited for | Longer conversations where full raw history is too expensive |

---

## ⚖️ Advantages & Limitations

<table>
<tr>
<td valign="top" width="50%">

### ✅ Advantages

- **Supports longer conversations** — older turns are compressed, not discarded
- **Lower context growth** — active prompt = summary + a few recent messages
- **Preserves important facts** — identity, preferences, projects, goals, decisions
- **Maintains recent detail** — the latest messages stay uncompressed
- **Simple architecture** — no vector database or retrieval pipeline needed

</td>
<td valign="top" width="50%">

### ⚠️ Limitations

- **Summary information loss** — details can be omitted, misunderstood, or over-compressed
- **Additional model calls** — summarization adds an extra LLM request
- **Summary drift** — repeated updates can gradually lose or distort details
- **Message-level retention** — the recent-message limit counts messages, not turns
- **No persistent storage** — summary and messages vanish at process end
- **No relevance retrieval** — older info survives only if the summarizer keeps it

</td>
</tr>
</table>

---

## 📁 Project Structure

```
3_summarization_memory/
├── README.md
├── __init__.py
├── agent.py        # Agent that combines summary and recent messages
├── demo.py         # Interactive command-line demonstration
├── memory.py       # Summary and recent-message state management
└── summarizer.py   # LLM-based conversation summarization
```

The agent also uses shared modules from the repository root:

```
common/
├── llm.py           # Creates the configured LangChain OpenAI chat model
└── token_tracker.py # Estimates and reports token usage
```

---

## 📋 Requirements

- 🐍 Python 3.9 or later
- 🔑 An OpenAI API key
- 📦 The dependencies listed in the repository's `requirements.txt`

The implementation uses [LangChain](https://www.langchain.com/) message types and `langchain-openai` for model access.

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

The shared LLM helper loads this variable with `python-dotenv` and creates a deterministic `gpt-4o-mini` chat model with `temperature=0`.

---

## ▶️ Running the Demo

Run the demo from the **repository root** using the package/import arrangement supported by the repository:

```bash
python -m 3_summarization_memory.demo
```

> ⚠️ **Note:** Because `3_summarization_memory` begins with a number, some Python environments do not accept it as a conventional module name. If module execution fails, rename the directory to a valid package identifier such as `summarization_memory`, update the relative imports if needed, and run `python -m summarization_memory.demo`.

The demo starts an interactive chat session. Enter `exit` or `quit` to stop.

```text
============================================================
SUMMARIZATION MEMORY
============================================================

Recent messages kept : 4
Type exit to quit
```

After each response, the demo prints token information for the retained messages and displays the current long-term summary.

### ⚠️ Demo configuration note

The current demo code initializes the agent with `max_recent_messages=3`:

```python
agent = SummarizationMemoryAgent(max_recent_messages=3)
```

However, its display text says `Recent messages kept : 4`. The implementation uses the constructor value, so the **active limit is 3 messages** unless the demo is updated to use `max_recent_messages=4`.

---

## 🐍 Usage as a Python Module

```python
from summarization_memory.agent import SummarizationMemoryAgent

agent = SummarizationMemoryAgent(max_recent_messages=4)

print(agent.chat("My name is Alex."))
print(agent.chat("I am building a customer-support bot."))
print(agent.chat("What project am I working on?"))
```

> If the source directory is kept under its current name (`3_summarization_memory`) and your environment supports the repository's import arrangement, use the corresponding package import. Otherwise, rename it to a valid Python package name such as `summarization_memory`.

Change the number of recent raw messages retained in the active context:

```python
agent = SummarizationMemoryAgent(max_recent_messages=6)
```

**Inspect the memory state:**

```python
memory = agent.get_memory()

print("Summary:")
print(memory.get_summary())

print("Recent messages:")
for message in memory.get_recent_messages():
    print(f"{message.type}: {message.content}")
```

**Clear the summary, recent messages, and token statistics:**

```python
agent.clear_memory()
```

---

## 🧠 Memory API

| Method | Description |
| --- | --- |
| `SummarizationMemory(max_recent_messages: int = 4)` | Creates an empty memory store with an empty summary and configurable recent-message count |
| `add_user_message(message: str)` | Appends a `HumanMessage` |
| `add_ai_message(message: str)` | Appends an `AIMessage` |
| `update_summary(summary: str)` | Replaces the stored long-term summary |
| `get_summary() -> str` | Returns the current long-term summary |
| `get_recent_messages() -> list` | Returns the newest `max_recent_messages` messages |
| `get_old_messages() -> list` | Returns messages older than the recent-message limit (empty if not exceeded) |
| `remove_old_messages()` | Removes messages outside the recent-message window |
| `clear()` | Resets both the summary and all stored messages |

---

## ✍️ Summarizer API

| Method | Description |
| --- | --- |
| `ConversationSummarizer()` | Creates a summarizer with the shared language model configuration |
| `summarize(existing_summary: str, messages) -> str` | Converts old messages to text and asks the LLM for an updated summary; returns only the summary text |

**The prompt instructs the model to preserve:**

- 🪪 User identity
- 🎯 User preferences
- 📌 Important facts
- 📁 Projects
- 🏁 Goals
- ✅ Decisions
- 🧩 Important context

---

## 🤖 Agent API

| Method | Description |
| --- | --- |
| `SummarizationMemoryAgent(max_recent_messages: int = 4)` | Creates an agent with a language model, a `SummarizationMemory`, a `ConversationSummarizer`, and a `TokenTracker` |
| `chat(user_message: str) -> str` | Adds the message, summarizes older messages when required, builds context, calls the LLM, stores the response, and returns it |
| `get_memory()` | Returns the complete `SummarizationMemory` object |
| `get_token_tracker()` | Returns the token tracker for current and cumulative usage |
| `clear_memory()` | Clears the summary and recent messages, then resets token statistics |

---

## 🔀 Message-Level Retention

`max_recent_messages` counts **individual messages**, not complete conversational turns. With `max_recent_messages=4`, the active raw memory can contain two user messages and two assistant responses.

Because summarization occurs **after** the new user message is added, the old-message set can end on a user message before the assistant has responded. This is normal for the current implementation — applications requiring complete user/assistant pairs may prefer pair-aware trimming.

---

## 📈 Complexity & Scaling

The active context size is bounded by the summary plus the configured number of recent messages — more scalable than sending the complete history on every request.

```mermaid
xychart-beta
    title "Context Size: Sequential vs. Sliding Window vs. Summarization"
    x-axis "Conversation Turn" [1, 5, 10, 15, 20, 25, 30]
    y-axis "Approx. Context Tokens" 0 --> 12000
    line [400, 2000, 4000, 6000, 8000, 10000, 12000]
    line [400, 800, 800, 800, 800, 800, 800]
    line [400, 900, 1100, 1150, 1200, 1220, 1250]
```

*(Top: unbounded sequential memory. Middle: fixed sliding window. Bottom: summarization — slightly larger than a pure window due to the summary, but far below unbounded growth.)*

> ⚠️ **Trade-off:** Summarization adds an **extra model call** whenever old messages exist. A conversation can therefore use *more total requests* than sequential or sliding-window memory, even though each regular response context is smaller. Total cost depends on summarization frequency, the length of old messages, the size of the generated summary, and token pricing.

---

## 🎚️ Choosing `max_recent_messages`

```mermaid
flowchart LR
    Q{How much verbatim<br/>recent context is needed?}
    Q -->|"Minimal"| S1["max_recent_messages = 2<br/>🔴 High summary frequency"]
    Q -->|"Typical"| S2["max_recent_messages = 4<br/>🟡 Moderate frequency"]
    Q -->|"More continuity"| S3["max_recent_messages = 6–10<br/>🟢 Lower frequency"]
    Q -->|"Max recent detail"| S4["Large values<br/>🟢 Lowest frequency, bigger prompts"]

    style S1 fill:#EF4444,color:#fff
    style S2 fill:#F59E0B,color:#111
    style S3 fill:#10B981,color:#fff
    style S4 fill:#10B981,color:#fff
```

| Value | Retained raw context | Summary frequency | Suitable for |
| --- | --- | --- | --- |
| `2` | One recent user/assistant exchange | High | Short commands and compact prompts |
| `4` | ~Two exchanges | Moderate | General prototypes and demonstrations |
| `6–10` | Several recent exchanges | Lower | Conversations needing more local continuity |
| Large values | More uncompressed context | Lower | Apps where recent detail is especially important |

A **smaller** value reduces the active prompt but triggers summarization more often. A **larger** value preserves more verbatim context but increases regular request size.

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
    Retrieval memory: [0.82, 0.85]
    Hybrid memory: [0.88, 0.9]
```

| Strategy | Retained information | Token efficiency | Long-range recall | Complexity | Typical use case |
| --- | --- | --- | --- | --- | --- |
| Sequential memory | Entire conversation | 🔴 Low | Excellent until context limits | 🟢 Low | Short demos and debugging |
| Sliding-window memory | Most recent messages | 🟡 Medium–High | 🔴 Low | 🟢 Low | Bounded recent context |
| **Summarization memory** | Summary + recent messages | 🟢 High | 🟡 Moderate (depends on summary quality) | 🟡 Medium | Long conversational sessions |
| Retrieval memory | Relevant historical items | 🟢 High | 🟢 High for retrievable info | 🔴 High | Knowledge-heavy assistants |
| Hybrid memory | Summary + recent turns + retrieved items | 🟢 High | 🟢 High | 🔴 High | Production conversational agents |

---

## 🎯 When to Use Summarization Memory

**Good fit when:**
- ✅ Conversations can grow longer than the model's practical context budget
- ✅ Older information should remain available in compressed form
- ✅ Recent messages need to remain verbatim
- ✅ You want a simpler alternative to vector retrieval
- ✅ You can tolerate occasional summarization calls and possible information loss

**Consider a different strategy when:**
- ❌ Only recent context matters → a **sliding window** is simpler and cheaper
- ❌ Specific older facts must be found accurately → **retrieval memory** is more reliable
- ❌ It's a short demo where exact history and simplicity beat token cost → **sequential memory**

---

## 🔢 Token Tracking

The agent uses the shared `TokenTracker` to report:

- 📏 Estimated tokens in the active context sent to the response model
- 📥 Cumulative input tokens reported by the model provider
- 📤 Cumulative output tokens reported by the model provider
- 📊 Total tokens, request count, and peak context size

```python
tracker = agent.get_token_tracker()

print(tracker.get_current_report())
print(tracker.get_total_report())
```

> ⚠️ **Note:** The tracker measures the *response* context built by the agent. Summarizer calls are separate model calls and may **not** be included in the agent's `TokenTracker` totals, because `ConversationSummarizer` invokes its model independently without recording that usage in the agent tracker.

---

## 🔒 Security & Privacy Notes

> ⚠️ The summary can contain personal information and important user context — treat it with the **same protection as the original conversation**.

- Do not send confidential or personally identifiable information to the model unless your application is designed and approved to handle it.
- Add retention policies, access controls, encryption, and persistent-storage safeguards before adapting this in-memory example for production.
- Call `clear_memory()` when a session should be discarded.
- Remember that clearing this object does **not** undo data already sent to an external model provider or stored in provider logs according to the provider's policies.

---

## 📄 License

Refer to the [root repository](https://github.com/paras160500/agent-memory-optimization) for the project's license and contribution guidelines.

---

## 🔗 References

| # | Resource |
| --- | --- |
| 1 | [Agent Memory Optimization repository](https://github.com/paras160500/agent-memory-optimization) |
| 2 | [`SummarizationMemory` implementation](https://github.com/paras160500/agent-memory-optimization/blob/main/3_summarization_memory/memory.py) |
| 3 | [`SummarizationMemoryAgent` implementation](https://github.com/paras160500/agent-memory-optimization/blob/main/3_summarization_memory/agent.py) |
| 4 | [`ConversationSummarizer` implementation](https://github.com/paras160500/agent-memory-optimization/blob/main/3_summarization_memory/summarizer.py) |
| 5 | [Summarization memory interactive demo](https://github.com/paras160500/agent-memory-optimization/blob/main/3_summarization_memory/demo.py) |
| 6 | [LangChain official website](https://www.langchain.com/) |
| 7 | [OpenAI text generation documentation](https://platform.openai.com/docs/guides/text-generation) |

<div align="center">

---

**⭐ Part of the [Agent Memory Optimization](https://github.com/paras160500/agent-memory-optimization) project ⭐**

</div>