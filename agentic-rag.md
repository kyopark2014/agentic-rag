# Agentic RAG

## 용어 정리

아래는 [7 Agentic RAG System Architectures](https://www.linkedin.com/posts/greg-coquillo_7-agentic-rag-system-architectures-ugcPost-7286098967434534912-F2Ek/?utm_source=share&utm_medium=member_android)에서 얘기한것인데 이해에 도움이 되어 정리해 봅니다.

- Agentic RAG Router

“Go through this door to find your answer.”



- Query Planning Agentic Rag

“Stay on the line, let me check in with a few experts before I can give you a complete answer.”


- Adaptive RAG

“Now that you ask the question this way, let me pull this special book from the library, along with other sources to answer you.”


- Agentic Corrective RAG

“Let me get rid of the information you don’t need before giving you a final answer”

- Self-Reflective RAG

“Let me check, does it even make sense? Hold on, let me see if anything needs correcting before answering you.”

- Speculative RAG

“I remember seeing this before from that source, but let me doublecheck before confirming for you.”

- Self Route Agentic RAG

“Sit back and relax, I know the best way to look for this answer. Be right back.”


## Multi agent에 대한 흥미로운 그림

[how multi-agent agentic RAG systems work](https://www.linkedin.com/posts/pavan-belagatti_lets-understand-how-multi-agent-agentic-activity-7286068649101008896-DmDB/?utm_source=share&utm_medium=member_android)을 참조합니다. 

- single agent에 비하여 복잡한 workflow와 여러 종류의 query 를 지원

- agent가 central orchestration을 수행하고 특정 목적에 최적화된 검색 agent들을 관리



![image](https://github.com/user-attachments/assets/319a13d3-01f6-44c2-92cb-4b11d3bf5c7c)


## Four Essential Agentic Design Patterns

[Four Essential Agentic Design Patterns](https://www.linkedin.com/posts/jillanisofttech_aiagents-futuretech-aipatterns-activity-7282996261299847168-5OlF/?utm_source=share&utm_medium=member_android)을 참조합니다.

4개의 중요 agent design 패턴들은 아래와 같습니다. 

1. Reflection Pattern

This pattern enables AI agents to learn from their experiences through:

- Initial response generation
- Self-analysis of outputs
- Continuous improvement through iteration

The reflection mechanism allows agents to enhance their decision-making capabilities over time, leading to more refined and accurate responses.

2. Planning Pattern

This strategic approach helps agents handle complex tasks by:
- Breaking down large tasks into manageable components
- Setting clear objectives and goals
- Implementing dynamic strategy adjustments
- Replanning when necessary

This pattern excels in scenarios requiring systematic problem-solving and adaptability.

3. Tool Use Pattern

The Tool Use Pattern extends AI capabilities by:

- Integrating with external tools and resources
- Managing multiple tool interactions
- Processing information from various sources

This framework allows agents to perform tasks beyond basic text generation, significantly expanding their practical applications.

4. Multi-Agent Pattern

This collaborative pattern enables:
- Coordination between specialized agents
- Efficient task delegation
- Combined expertise in complex problems

Perfect for scenarios requiring diverse skills, like software development or project management.

![image](https://github.com/user-attachments/assets/6b72494a-57bb-4cc8-a77a-94e5ecccd518)
