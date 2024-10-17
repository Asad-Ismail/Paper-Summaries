## Agents

Below is summary with some thoughts/addition on the excellent review paper **A Survey on Large Language Model based Autonomous
Agents** [Paper](https://arxiv.org/pdf/2308.11432)

### Definition of an Autonomous Agent

> "An autonomous agent is a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future."
> 
> — Franklin and Graesser (1997)

### Reinforcement Learning Vs LLM agents:

### 1. Learning Paradigms
   - **Reinforcement Learning (RL)**: RL involves learning through direct interaction and feedback from the environment. Agents adjust their strategies based on rewards or penalties, optimizing their actions over time to maximize cumulative rewards. This approach is inherently suited for tasks requiring exploration, adaptation, and continuous improvement within dynamic environments.
   - **Large Language Models (LLMs)**: LLMs are trained on vast datasets through supervised learning, focusing on predicting the next word in a sequence. They don’t interact with live environments but instead derive insights from large text corpora. This enables them to generate coherent and contextually relevant language, but their static training may limit adaptability in scenarios requiring real-time learning.

### 2. Decision-Making Frameworks
   - **RL Agents**: These agents make sequential decisions within a Markov Decision Process (MDP) framework, where each action affects the future state and rewards. This structured approach allows RL agents to optimize strategies for long-term benefits and handle complex dependencies over time.
   - **LLMs**: LLMs lack a structured decision-making framework. They generate outputs based on probability distributions rather than explicit planning mechanisms, which can lead to significant limitations in scenarios requiring accurate and reliable planning. As a result, LLM-generated plans can sometimes be incomplete or outright incorrect, especially for complex, multi-step tasks.

### 3. Knowledge and Domain Adaptability
   - **World Knowledge**: LLM-based agents possess extensive internalized knowledge, enabling them to generalize across diverse topics without domain-specific training. This makes them highly versatile for generating information or simulating general knowledge across many fields.
   - **Domain-Specific Learning in RL**: RL agents typically excel in specialized tasks where they can interact with the environment. However, they may struggle with generalization across different domains without targeted retraining, as they rely heavily on specific reward structures and experiences from their training environments.

### 4. Reliability and Planning Accuracy
   - **LLMs**:  LLMs are powerful for generating ideas and simulating dialogue, they may lack reliability in decision-making and planning. Their plans can sometimes be entirely wrong or implausible due to their reliance on statistical associations rather than factual correctness. Without an inherent model of the environment or ability to verify actions, LLMs may produce errors, particularly for complex or sequential tasks.
   - **RL Agents**: RL agents are generally more reliable in planning for specific tasks within familiar environments, as they can iteratively refine their strategies based on feedback. However, they may lack flexibility for tasks outside their trained domains, limiting their use in general-purpose planning.


We will focus mostly on LLM agents RL agents will follow its seperate series of posts

### Construction of LLM agents
In order to construct the LLM agensts, there are two significant aspects, that is,
1. Which architecture should be designed to better
use LLMs
2. Give the designed architecture,
how to enable the agent to acquire capabilities for
accomplishing specific tasks.

When comparing LLM-based autonomous
agents to traditional machine learning, designing
the agent architecture is analogous to determining
the network structure, while the agent capability
acquisition is similar to learning the network parameters.

#### Agent Architectures

We need to provide LLMs with different "modules" to enance their capabilities and act like an agent in an environment below provides 

**Profiling Module:**

The profiling module aims
to indicate the profiles of the agent roles, which
are usually written into the prompt to influence the
LLM behaviors. Agent profiles typically encompass basic information such as age, gender, and
career as well as psychology information, reflecting the personalities of the agent, and social
information, detailing the relationships between
agents. The choice of information to profile the
agent is largely determined by the specific application scenarios.
The best practice for now is to get the profile information from real world dataset like age, gender, personal traits and movie preferences, if it is not available make some handcrafted profiles and use them as few shot examples and then use LLMs for creating any new agent profiles that might be needed

**Memory Module:**

Memory module stores information perceived from the environment and leverages
the recorded memories to facilitate future actions.
The memory module can help the agent to accumulate experiences, self-evolve, and behave in a more
consistent, reasonable, and effective manner. Two commonly used memories are 

1. Short Term Memory

2. Long Term Memory

**Unified Memory**

This structure only simulates the human shot-term memory, which is usually
realized by in-context learning, and the memory information is directly written into the prompts e.g CALYPSO is an agent designed for the game Dungeons & Dragons, which
can assist Dungeon Masters in the creation and narration of stories. Its short-term memory is built
upon scene descriptions, monster information, and
previous summaries. The drawback of short term memory is limitation of context window
of LLMs, it’s hard to put all memories into prompt,
which may degrade the performance of agents.This
method has high requirements on the window length
of LLMs and the ability to handle long contexts.


**Hybrid Memory**

Hybrid Memory explicitly models the "maybe" human short-term and long-term memories. The short-term memory temporarily buffers recent perceptions, while long-term memory consolidates important information over time. e.g AgentSims  implements a hybrid memory architecture. The information provided in the prompt can be considered as short-term
In order to enhance the storage capacity of memory, the authors propose a long-term
memory system that utilizes a vector database, facilitating efficient storage and retrieval. Specifically, the agent’s daily memories are encoded as
embeddings and stored in the vector database. If
the agent needs to recall its previous memories, the
long-term memory system retrieves relevant information using embedding similarities. This process
can improve the consistency of the agent’s behavior. 

**Memory Formats**

1. Natural Language

   Firstly, the memory information can be expressed
   in a flexible and understandable manner. Moreover,
   it retains rich semantic information that can provide
   comprehensive signals to guide agent behaviors. In
   the previous work, Reflexion [12] stores experiential feedback in natural language within a sliding
   window.

2. Embeddings

   Memory data is stored in embeddings vectors, e.g ChatDev
   encodes dialogue history into vectors for retrieval.



3. Database


   In this format, memory information is stored in databases, allowing the agent
   to manipulate memories efficiently and comprehensively. For example, ChatDB [40] uses a database as
   a symbolic memory module. The agent can utilize
   SQL statements to precisely add, delete, and revise the memory information. In DB-GPT [41], the
   memory module is constructed based on a database.
   To more intuitively operate the memory information, the agents are fine-tuned to understand and
   execute SQL queries, enabling them to interact with
   databases using natural language directly.

4. Stuctured List


   In this format, memory information is organized into lists, and the semantic of
   memory can be conveyed in an efficient and concise
   manner. For instance, GITM stores action lists
   for sub-goals in a hierarchical tree structure
   It utilizes a key-value list structure. In this structure, the keys are represented by
   embedding vectors, while the values consist of raw
   natural languages.


### Memory Operations
Memory Reading:

